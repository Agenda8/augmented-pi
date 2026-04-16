import dataclasses
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass
class DepthGuidedAlignConfig:
    """Config for a simple depth-based near-object alignment controller."""

    enabled: bool = False
    once_per_episode: bool = True

    roi_top_ratio: float = 0.10
    roi_bottom_ratio: float = 0.95
    roi_left_ratio: float = 0.10
    roi_right_ratio: float = 0.90

    mask_bottom_start_ratio: float = 0.55
    mask_depth_value_threshold: float = 0.9
    mask_min_pixels: int = 60

    # Pixel anchor where we expect the object (e.g., between gripper fingers in wrist view).
    # (0, 0) is top-left and (1, 1) is bottom-right.
    align_target_u_ratio: float = 0.50
    align_target_v_ratio: float = 0.78
    align_target_depth: float = 0.85

    detect_percentile: float = 5.0
    detect_object_depth_threshold: float = 0.9
    min_near_pixels: int = 80
    trigger_target_radius_px: float = 80.0

    align_pixel_tolerance: float = 6.0
    align_depth_tolerance: float = 0.01
    align_hold_steps: int = 3
    max_align_steps: int = 40
    close_gripper_steps: int = 4

    x_from_v_gain: float = -1
    y_from_u_gain: float = -1
    z_from_depth_gain: float = -1
    z_bias: float = 0.0
    max_translation_step: float = 0.1

    gripper_open_value: float = -1.0
    gripper_close_value: float = 1.0


@dataclasses.dataclass
class DepthObject:
    cx: float
    cy: float
    near_depth: float
    pixel_count: int


@dataclasses.dataclass
class DepthFrameAnalysis:
    target_found: bool
    should_trigger: bool
    target_anchor: Tuple[float, float]
    target_cx: Optional[float] = None
    target_cy: Optional[float] = None
    center_dist: Optional[float] = None
    target_depth: Optional[float] = None
    depth_error: Optional[float] = None
    depth_gap_abs: Optional[float] = None
    is_aligned: bool = False
    pixel_count: int = 0
    near_mask: Optional[np.ndarray] = None
    gripper_mask: Optional[np.ndarray] = None


class DepthGuidedAligner:
    """
    Small state-machine controller used at evaluation time:
    1) wait for a near depth object around the reference point;
    2) align its centroid to gripper center via XY translations;
    3) close gripper for a few steps;
    4) hand control back to VLA.

    This is intentionally lightweight and heuristic-based for quick validation.
    """

    def __init__(self, config: DepthGuidedAlignConfig):
        self._config = config
        self.reset_episode()

    def reset_episode(self) -> None:
        self._mode = "idle"
        self._used_once = False
        self._align_steps = 0
        self._align_hold_steps = 0
        self._close_steps = 0
        self._initial_gripper_mask = None

    def get_mode(self) -> str:
        return self._mode

    def get_control_action(self, depth: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if not self._config.enabled:
            return None, None

        if self._mode == "idle" and self._config.once_per_episode and self._used_once:
            return None, None

        detected_object, target_point, _ = self._extract_object(depth)

        if self._mode == "idle":
            if detected_object is None or target_point is None:
                return None, None
            if self._should_trigger(detected_object, target_point):
                self._mode = "align"
                self._align_steps = 0
                self._align_hold_steps = 0
                return self._build_alignment_action(detected_object, target_point), "depth align triggered"
            return None, None

        if self._mode == "align":
            self._align_steps += 1

            if detected_object is None or target_point is None:
                if self._align_steps >= self._config.max_align_steps:
                    self._mode = "close"
                    self._close_steps = 0
                    return self._close_gripper_action(), "depth align timed out, close gripper"
                return self._hold_open_action(), None

            action = self._build_alignment_action(detected_object, target_point)
            if self._is_object_aligned(detected_object, target_point):
                self._align_hold_steps += 1
            else:
                self._align_hold_steps = 0

            if self._align_hold_steps >= self._config.align_hold_steps:
                self._mode = "close"
                self._close_steps = 0
                return self._close_gripper_action(), "depth align done, close gripper"

            if self._align_steps >= self._config.max_align_steps:
                self._mode = "close"
                self._close_steps = 0
                return self._close_gripper_action(), "depth align max steps reached, close gripper"

            return action, None

        if self._mode == "close":
            self._close_steps += 1
            action = self._close_gripper_action()
            if self._close_steps >= self._config.close_gripper_steps:
                self._mode = "idle"
                self._align_steps = 0
                self._align_hold_steps = 0
                if self._config.once_per_episode:
                    self._used_once = True
                return action, "depth align finished, back to VLA"
            return action, None

        return None, None

    def analyze_depth_frame(self, depth: Optional[np.ndarray], include_mask: bool = False) -> DepthFrameAnalysis:
        detected_object, target_point, near_mask = self._extract_object(depth, include_mask=include_mask)
        target_depth = float(self._config.align_target_depth)
        if detected_object is None:
            if target_point is None:
                target_point = (0.0, 0.0)
            return DepthFrameAnalysis(
                target_found=False,
                should_trigger=False,
                target_anchor=target_point,
                target_depth=target_depth,
                pixel_count=0,
                near_mask=near_mask,
                gripper_mask=self._initial_gripper_mask if include_mask else None,
            )

        center_dist = float(np.hypot(detected_object.cx - target_point[0], detected_object.cy - target_point[1]))
        depth_error = self._compute_depth_error(detected_object)
        depth_gap_abs = abs(depth_error)
        is_aligned = self._is_object_aligned(detected_object, target_point)

        return DepthFrameAnalysis(
            target_found=True,
            should_trigger=self._should_trigger(detected_object, target_point),
            target_anchor=target_point,
            target_cx=detected_object.cx,
            target_cy=detected_object.cy,
            center_dist=center_dist,
            target_depth=target_depth,
            depth_error=depth_error,
            depth_gap_abs=depth_gap_abs,
            is_aligned=is_aligned,
            pixel_count=detected_object.pixel_count,
            near_mask=near_mask,
            gripper_mask=self._initial_gripper_mask if include_mask else None,
        )

    def _extract_object(
        self,
        depth: Optional[np.ndarray],
        include_mask: bool = False,
    ) -> Tuple[Optional[DepthObject], Optional[Tuple[float, float]], Optional[np.ndarray]]:
        if depth is None:
            return None, None, None

        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None, None, None

        if self._initial_gripper_mask is None:
            self._initial_gripper_mask = self._build_initial_gripper_mask(arr)

        h, w = arr.shape
        target_point = self._compute_target_point(w, h)

        y0 = int(h * self._config.roi_top_ratio)
        y1 = int(h * self._config.roi_bottom_ratio)
        x0 = int(w * self._config.roi_left_ratio)
        x1 = int(w * self._config.roi_right_ratio)
        if y1 <= y0 or x1 <= x0:
            return None, target_point, None

        roi = arr[y0:y1, x0:x1]
        finite_mask = np.isfinite(roi)
        if self._initial_gripper_mask is not None:
            finite_mask &= ~self._initial_gripper_mask[y0:y1, x0:x1]
        if not np.any(finite_mask):
            return None, target_point, None

        valid = roi[finite_mask]
        if valid.size < self._config.min_near_pixels:
            return None, target_point, None

        if self._config.detect_object_depth_threshold is not None:
            detect_threshold = float(self._config.detect_object_depth_threshold)
        else:
            detect_threshold = float(np.percentile(valid, self._config.detect_percentile))
        near_mask = finite_mask & (roi <= detect_threshold)
        near_mask = self._largest_connected_component(near_mask)

        ys, xs = np.nonzero(near_mask)
        if ys.size < self._config.min_near_pixels:
            return None, target_point, None

        # Weighted center emphasizes the closest pixels.
        weights = (detect_threshold - roi[ys, xs]).astype(np.float64) + 1e-6
        cx = float(np.average(xs + x0, weights=weights))
        cy = float(np.average(ys + y0, weights=weights))
        object_depth = float(np.average(roi[ys, xs], weights=weights))

        full_mask = None
        if include_mask:
            full_mask = np.zeros((h, w), dtype=bool)
            full_mask[y0:y1, x0:x1] = near_mask

        return DepthObject(
            cx=cx,
            cy=cy,
            near_depth=object_depth,
            pixel_count=int(ys.size),
        ), target_point, full_mask

    def _build_initial_gripper_mask(self, depth_2d: np.ndarray) -> np.ndarray:
        h, w = depth_2d.shape
        mask = np.zeros((h, w), dtype=bool)

        y0 = int(h * self._config.mask_bottom_start_ratio)
        y0 = int(np.clip(y0, 0, h - 1))

        # Do not crop horizontally: build gripper mask from the full width.
        x0 = 0
        x1 = w

        region = depth_2d[y0:, x0:x1]
        valid = region[np.isfinite(region)]
        if valid.size < self._config.mask_min_pixels:
            return mask

        cutoff = float(self._config.mask_depth_value_threshold)
        candidates = np.isfinite(depth_2d) & (depth_2d <= cutoff)

        bottom_gate = np.zeros_like(mask)
        bottom_gate[y0:, x0:x1] = True
        candidates &= bottom_gate

        if np.count_nonzero(candidates) < self._config.mask_min_pixels:
            return mask

        connected = self._connected_to_bottom(candidates)
        if np.count_nonzero(connected) < self._config.mask_min_pixels:
            return mask
        return connected

    def _connected_to_bottom(self, candidates: np.ndarray) -> np.ndarray:
        h, w = candidates.shape
        visited = np.zeros_like(candidates, dtype=bool)
        stack = []

        for x in range(w):
            if candidates[h - 1, x]:
                visited[h - 1, x] = True
                stack.append((h - 1, x))

        while stack:
            y, x = stack.pop()
            if y > 0 and candidates[y - 1, x] and not visited[y - 1, x]:
                visited[y - 1, x] = True
                stack.append((y - 1, x))
            if y < h - 1 and candidates[y + 1, x] and not visited[y + 1, x]:
                visited[y + 1, x] = True
                stack.append((y + 1, x))
            if x > 0 and candidates[y, x - 1] and not visited[y, x - 1]:
                visited[y, x - 1] = True
                stack.append((y, x - 1))
            if x < w - 1 and candidates[y, x + 1] and not visited[y, x + 1]:
                visited[y, x + 1] = True
                stack.append((y, x + 1))

        return visited

    def _largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        best_count = 0
        best_pixels = []

        for y0 in range(h):
            for x0 in range(w):
                if not mask[y0, x0] or visited[y0, x0]:
                    continue

                stack = [(y0, x0)]
                visited[y0, x0] = True
                component_pixels = []

                while stack:
                    y, x = stack.pop()
                    component_pixels.append((y, x))

                    if y > 0 and mask[y - 1, x] and not visited[y - 1, x]:
                        visited[y - 1, x] = True
                        stack.append((y - 1, x))
                    if y < h - 1 and mask[y + 1, x] and not visited[y + 1, x]:
                        visited[y + 1, x] = True
                        stack.append((y + 1, x))
                    if x > 0 and mask[y, x - 1] and not visited[y, x - 1]:
                        visited[y, x - 1] = True
                        stack.append((y, x - 1))
                    if x < w - 1 and mask[y, x + 1] and not visited[y, x + 1]:
                        visited[y, x + 1] = True
                        stack.append((y, x + 1))

                if len(component_pixels) > best_count:
                    best_count = len(component_pixels)
                    best_pixels = component_pixels

        largest = np.zeros_like(mask, dtype=bool)
        for y, x in best_pixels:
            largest[y, x] = True
        return largest

    def _compute_target_point(self, width: int, height: int) -> Tuple[float, float]:
        u = float(np.clip(self._config.align_target_u_ratio, 0.0, 1.0))
        v = float(np.clip(self._config.align_target_v_ratio, 0.0, 1.0))
        return ((width - 1) * u, (height - 1) * v)

    def _should_trigger(self, detected_object: DepthObject, target_point: Tuple[float, float]) -> bool:
        dx = detected_object.cx - target_point[0]
        dy = detected_object.cy - target_point[1]
        center_dist = float(np.hypot(dx, dy))

        return (
            detected_object.pixel_count >= self._config.min_near_pixels
            and center_dist <= self._config.trigger_target_radius_px
        )

    def _is_object_aligned(self, detected_object: DepthObject, target_point: Tuple[float, float]) -> bool:
        dx = detected_object.cx - target_point[0]
        dy = detected_object.cy - target_point[1]
        center_dist = float(np.hypot(dx, dy))
        depth_error = abs(self._compute_depth_error(detected_object))
        return (
            center_dist <= self._config.align_pixel_tolerance
            and depth_error <= self._config.align_depth_tolerance
        )

    def _compute_depth_error(self, detected_object: DepthObject) -> float:
        desired_depth = float(self._config.align_target_depth)
        return float(detected_object.near_depth - desired_depth)

    def _build_alignment_action(self, detected_object: DepthObject, target_point: Tuple[float, float]) -> np.ndarray:
        # u: horizontal axis, v: vertical axis in image coordinates
        u_err = (detected_object.cx - target_point[0]) / max(target_point[0], 1.0)
        v_err = (detected_object.cy - target_point[1]) / max(target_point[1], 1.0)
        depth_err = self._compute_depth_error(detected_object)

        dx = float(np.clip(self._config.x_from_v_gain * v_err, -self._config.max_translation_step, self._config.max_translation_step))
        dy = float(np.clip(self._config.y_from_u_gain * u_err, -self._config.max_translation_step, self._config.max_translation_step))
        dz_cmd = self._config.z_bias + self._config.z_from_depth_gain * depth_err
        dz = float(np.clip(dz_cmd, -self._config.max_translation_step, self._config.max_translation_step))

        action = np.zeros(7, dtype=np.float32)
        action[0] = dx
        action[1] = dy
        action[2] = dz
        action[6] = self._config.gripper_open_value
        return action

    def _hold_open_action(self) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[6] = self._config.gripper_open_value
        return action

    def _close_gripper_action(self) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[6] = self._config.gripper_close_value
        return action
