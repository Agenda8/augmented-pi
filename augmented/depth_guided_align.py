import dataclasses
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass
class DepthGuidedAlignConfig:
    """Config for a simple depth-based near-object alignment controller."""

    enabled: bool = False

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
    align_target_depth: float = 0.87

    detect_percentile: float = 5.0
    detect_object_depth_threshold: float = 0.9
    min_near_pixels: int = 80
    trigger_target_radius_px: float = 50.0

    align_x_tolerance_px: float = 10.0
    align_y_tolerance_px: float = 10.0
    align_depth_tolerance: float = 0.02
    align_hold_steps: int = 1
    max_align_steps: int = 50
    # Prevent immediate re-trigger after a failed/timeout alignment attempt.
    retry_cooldown_steps_after_failure: int = 30
    close_gripper_steps: int = 4

    # Only allow depth recognition / triggering when recent commanded gripper
    # actions have stayed open for a history window (useful for multi-object pick tasks).
    require_open_gripper_for_detection: bool = True
    open_gripper_cmd_history_steps: int = 20

    # Pose-guided mapping from image errors to base-frame XY translation.
    # When enabled, the controller uses eef orientation + wrist camera mount to
    # compute per-step XY displacement in the robot base frame.
    use_pose_guided_xy: bool = True
    pose_xy_gain: float = 1.0
    pose_min_depth: float = 0.05
    pose_default_camera_fovy_deg: float = 45.0
    pose_u_to_cam_x_sign: float = 1.0
    pose_v_to_cam_y_sign: float = 1.0
    pose_processed_image_is_180_rotated: bool = True

    # Motion conversion gains from image/depth errors to robot translation.
    x_from_v_gain: float = 0.5
    y_from_u_gain: float = 1
    z_from_depth_gain: float = -3
    z_bias: float = 0.0
    max_translation_step: float = 0.8

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
    can_run_detection_now: bool
    gripper_cmd: Optional[float]
    target_anchor: Tuple[float, float]
    target_cx: Optional[float] = None
    target_cy: Optional[float] = None
    x_error: Optional[float] = None
    y_error: Optional[float] = None
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
        self._camera_fovy_deg = float(config.pose_default_camera_fovy_deg)
        self._eef_to_cam_rot = None
        self.reset_episode()

    def reset_episode(self) -> None:
        self._mode = "idle"
        self._align_steps = 0
        self._align_hold_steps = 0
        self._close_steps = 0
        self._retry_cooldown_steps = 0
        self._initial_gripper_mask = None
        self._eef_to_cam_rot = None
        self._gripper_open_history = []

    def get_mode(self) -> str:
        return self._mode

    def get_control_action(
        self,
        depth: Optional[np.ndarray],
        eef_pos: Optional[np.ndarray] = None,
        eef_quat: Optional[np.ndarray] = None,
        camera_rot_base: Optional[np.ndarray] = None,
        camera_fovy_deg: Optional[float] = None,
        gripper_cmd: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if not self._config.enabled:
            return None, None

        self._update_pose_context(eef_quat=eef_quat, camera_rot_base=camera_rot_base, camera_fovy_deg=camera_fovy_deg)
        self._update_gripper_open_history(gripper_cmd)

        if self._mode == "idle":
            if self._retry_cooldown_steps > 0:
                self._retry_cooldown_steps -= 1
                return None, None

            if not self._can_run_detection_now():
                return None, None

            detected_object, target_point, _ = self._extract_object(depth)
            depth_shape = self._get_depth_shape(depth)
            if detected_object is None or target_point is None:
                return None, None
            if self._should_trigger(detected_object, target_point):
                self._mode = "align"
                self._align_steps = 0
                self._align_hold_steps = 0
                return (
                    self._build_alignment_action(
                        detected_object,
                        target_point,
                        depth_shape=depth_shape,
                        eef_pos=eef_pos,
                        eef_quat=eef_quat,
                    ),
                    "depth align triggered",
                )
            return None, None

        if self._mode == "align":
            self._align_steps += 1
            detected_object, target_point, _ = self._extract_object(depth)
            depth_shape = self._get_depth_shape(depth)

            if detected_object is None or target_point is None:
                if self._align_steps >= self._config.max_align_steps:
                    self._mode = "idle"
                    self._align_steps = 0
                    self._align_hold_steps = 0
                    self._start_retry_cooldown()
                    return None, "depth align timed out, back to VLA"
                return self._hold_open_action(), None

            action = self._build_alignment_action(
                detected_object,
                target_point,
                depth_shape=depth_shape,
                eef_pos=eef_pos,
                eef_quat=eef_quat,
            )
            if self._is_object_aligned(detected_object, target_point):
                self._align_hold_steps += 1
            else:
                self._align_hold_steps = 0

            if self._align_hold_steps >= self._config.align_hold_steps:
                self._mode = "close"
                self._close_steps = 0
                return self._close_gripper_action(), "depth align done, close gripper"

            if self._align_steps >= self._config.max_align_steps:
                self._mode = "idle"
                self._align_steps = 0
                self._align_hold_steps = 0
                self._start_retry_cooldown()
                return None, "depth align max steps reached, back to VLA"

            return action, None

        if self._mode == "close":
            self._close_steps += 1
            action = self._close_gripper_action()
            if self._close_steps >= self._config.close_gripper_steps:
                self._mode = "idle"
                self._align_steps = 0
                self._align_hold_steps = 0
                return action, "depth align finished, back to VLA"
            return action, None

        return None, None

    def analyze_depth_frame(
        self,
        depth: Optional[np.ndarray],
        include_mask: bool = False,
        gripper_cmd: Optional[float] = None,
    ) -> DepthFrameAnalysis:
        can_run_detection_now = self._can_run_detection_now()
        gripper_cmd_value = None
        if gripper_cmd is not None and np.isfinite(gripper_cmd):
            gripper_cmd_value = float(gripper_cmd)

        # History is updated in get_control_action once per control step.
        # Keep analysis read-only to avoid counting the same step twice.
        if self._mode == "idle" and (self._retry_cooldown_steps > 0 or not self._can_run_detection_now()):
            target_depth = float(self._config.align_target_depth)
            target_point = (0.0, 0.0)
            depth_shape = self._get_depth_shape(depth)
            if depth_shape is not None:
                h, w = depth_shape
                target_point = self._compute_target_point(w, h)
            return DepthFrameAnalysis(
                target_found=False,
                should_trigger=False,
                can_run_detection_now=can_run_detection_now,
                gripper_cmd=gripper_cmd_value,
                target_anchor=target_point,
                target_depth=target_depth,
                pixel_count=0,
                near_mask=None,
                gripper_mask=self._initial_gripper_mask if include_mask else None,
            )

        detected_object, target_point, near_mask = self._extract_object(depth, include_mask=include_mask)
        target_depth = float(self._config.align_target_depth)
        if detected_object is None:
            if target_point is None:
                target_point = (0.0, 0.0)
            return DepthFrameAnalysis(
                target_found=False,
                should_trigger=False,
                can_run_detection_now=can_run_detection_now,
                gripper_cmd=gripper_cmd_value,
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
        x_error = float(detected_object.cx - target_point[0])
        y_error = float(detected_object.cy - target_point[1])

        return DepthFrameAnalysis(
            target_found=True,
            should_trigger=self._should_trigger(detected_object, target_point),
            can_run_detection_now=can_run_detection_now,
            gripper_cmd=gripper_cmd_value,
            target_anchor=target_point,
            target_cx=detected_object.cx,
            target_cy=detected_object.cy,
            x_error=x_error,
            y_error=y_error,
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

    def _update_gripper_open_history(self, gripper_cmd: Optional[float]) -> None:
        if not self._config.require_open_gripper_for_detection:
            return

        window = max(1, int(self._config.open_gripper_cmd_history_steps))
        is_open = False

        if gripper_cmd is not None and np.isfinite(gripper_cmd):
            value = float(gripper_cmd)
            is_open = value <= 0

        self._gripper_open_history.append(bool(is_open))
        if len(self._gripper_open_history) > window:
            self._gripper_open_history = self._gripper_open_history[-window:]

    def _can_run_detection_now(self) -> bool:
        if not self._config.require_open_gripper_for_detection:
            return True

        window = max(1, int(self._config.open_gripper_cmd_history_steps))
        if len(self._gripper_open_history) < window:
            return False
        return all(self._gripper_open_history[-window:])

    def _start_retry_cooldown(self) -> None:
        self._retry_cooldown_steps = max(0, int(self._config.retry_cooldown_steps_after_failure))
        # Also clear history so a fresh open-command window is required.
        self._gripper_open_history = []

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
        depth_error = abs(self._compute_depth_error(detected_object))
        return (
            abs(dx) <= self._config.align_x_tolerance_px
            and abs(dy) <= self._config.align_y_tolerance_px
            and depth_error <= self._config.align_depth_tolerance
        )

    def _compute_depth_error(self, detected_object: DepthObject) -> float:
        desired_depth = float(self._config.align_target_depth)
        return float(detected_object.near_depth - desired_depth)

    def _build_alignment_action(
        self,
        detected_object: DepthObject,
        target_point: Tuple[float, float],
        depth_shape: Optional[Tuple[int, int]],
        eef_pos: Optional[np.ndarray],
        eef_quat: Optional[np.ndarray],
    ) -> np.ndarray:
        # u: horizontal axis, v: vertical axis in image coordinates
        u_err = (detected_object.cx - target_point[0]) / max(target_point[0], 1.0)
        v_err = (detected_object.cy - target_point[1]) / max(target_point[1], 1.0)
        depth_err = self._compute_depth_error(detected_object)

        pose_xy = self._compute_pose_guided_xy_delta(
            detected_object=detected_object,
            target_point=target_point,
            depth_shape=depth_shape,
            eef_quat=eef_quat,
        )
        if pose_xy is None:
            dx = float(
                np.clip(
                    self._config.x_from_v_gain * v_err,
                    -self._config.max_translation_step,
                    self._config.max_translation_step,
                )
            )
            dy = float(
                np.clip(
                    self._config.y_from_u_gain * u_err,
                    -self._config.max_translation_step,
                    self._config.max_translation_step,
                )
            )
        else:
            dx, dy = pose_xy

        dz_cmd = self._config.z_bias + self._config.z_from_depth_gain * depth_err
        dz = float(np.clip(dz_cmd, -self._config.max_translation_step, self._config.max_translation_step))

        action = np.zeros(7, dtype=np.float32)
        action[0] = dx
        action[1] = dy
        action[2] = dz
        action[6] = self._config.gripper_open_value
        return action

    def _update_pose_context(
        self,
        eef_quat: Optional[np.ndarray],
        camera_rot_base: Optional[np.ndarray],
        camera_fovy_deg: Optional[float],
    ) -> None:
        if camera_fovy_deg is not None:
            fovy = float(camera_fovy_deg)
            if np.isfinite(fovy) and fovy > 1e-3:
                self._camera_fovy_deg = fovy

        if not self._config.use_pose_guided_xy:
            return
        if eef_quat is None or camera_rot_base is None:
            return

        eef_rot_base = self._quat_xyzw_to_rotmat(eef_quat)
        cam_rot_base = np.asarray(camera_rot_base, dtype=np.float64)
        if cam_rot_base.shape != (3, 3):
            return
        self._eef_to_cam_rot = eef_rot_base.T @ cam_rot_base

    def _compute_pose_guided_xy_delta(
        self,
        detected_object: DepthObject,
        target_point: Tuple[float, float],
        depth_shape: Optional[Tuple[int, int]],
        eef_quat: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float]]:
        if not self._config.use_pose_guided_xy:
            return None
        if self._eef_to_cam_rot is None or eef_quat is None:
            return None
        if depth_shape is None or len(depth_shape) != 2:
            return None

        height, width = depth_shape
        if height <= 1 or width <= 1:
            return None

        fx, fy = self._compute_focal_lengths(width=width, height=height)
        if fx <= 1e-6 or fy <= 1e-6:
            return None

        u_err_px = float(detected_object.cx - target_point[0])
        v_err_px = float(detected_object.cy - target_point[1])

        if self._config.pose_processed_image_is_180_rotated:
            u_err_px = -u_err_px
            v_err_px = -v_err_px

        object_depth = max(float(detected_object.near_depth), float(self._config.pose_min_depth))
        delta_cam_x = self._config.pose_u_to_cam_x_sign * (u_err_px * object_depth / fx)
        delta_cam_y = self._config.pose_v_to_cam_y_sign * (v_err_px * object_depth / fy)
        delta_cam = np.array([delta_cam_x, delta_cam_y, 0.0], dtype=np.float64)

        eef_rot_base = self._quat_xyzw_to_rotmat(eef_quat)
        cam_rot_base = eef_rot_base @ self._eef_to_cam_rot
        delta_base = cam_rot_base @ delta_cam

        gain = float(self._config.pose_xy_gain)
        dx = float(
            np.clip(
                gain * delta_base[0],
                -self._config.max_translation_step,
                self._config.max_translation_step,
            )
        )
        dy = float(
            np.clip(
                gain * delta_base[1],
                -self._config.max_translation_step,
                self._config.max_translation_step,
            )
        )
        return dx, dy

    def _compute_focal_lengths(self, width: int, height: int) -> Tuple[float, float]:
        fovy = float(np.clip(self._camera_fovy_deg, 1.0, 179.0))
        half_fovy_rad = np.deg2rad(fovy) * 0.5
        fy = 0.5 * float(height) / np.tan(half_fovy_rad)
        fx = fy * (float(width) / max(float(height), 1.0))
        return float(fx), float(fy)

    def _get_depth_shape(self, depth: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
        if depth is None:
            return None
        arr = np.asarray(depth)
        if arr.ndim == 3 and arr.shape[-1] >= 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None
        return int(arr.shape[0]), int(arr.shape[1])

    def _quat_xyzw_to_rotmat(self, quat_xyzw: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
        if quat.size != 4:
            return np.eye(3, dtype=np.float64)

        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            return np.eye(3, dtype=np.float64)

        x, y, z, w = quat / norm

        xx = x * x
        yy = y * y
        zz = z * z
        ww = w * w
        xy = x * y
        xz = x * z
        yz = y * z
        xw = x * w
        yw = y * w
        zw = z * w

        return np.array(
            [
                [ww + xx - yy - zz, 2.0 * (xy - zw), 2.0 * (xz + yw)],
                [2.0 * (xy + zw), ww - xx + yy - zz, 2.0 * (yz - xw)],
                [2.0 * (xz - yw), 2.0 * (yz + xw), ww - xx - yy + zz],
            ],
            dtype=np.float64,
        )

    def _hold_open_action(self) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[6] = self._config.gripper_open_value
        return action

    def _close_gripper_action(self) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[6] = self._config.gripper_close_value
        return action
