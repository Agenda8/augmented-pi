import csv
import dataclasses
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tyro

from augmented.depth_guided_align import DepthFrameAnalysis, DepthGuidedAlignConfig, DepthGuidedAligner

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    action_path: str
    task_suite_name: str = "libero_10"
    task_id: int = 0
    initial_state_idx: int = 0
    seed: int = 7
    num_steps_wait: int = 10
    max_steps: int = 520

    # Stop replay once depth align would trigger. This replays only pre-align actions.
    stop_before_align: bool = False

    save_trace: bool = True
    trace_out_dir: str = "data/libero/depth_align_trace_replay"
    save_align_frames: bool = True
    # If true, only save frames while align is active (or on trigger events).
    save_only_active_align_frames: bool = True


def main(args: Args) -> None:
    np.random.seed(args.seed)

    action_path = pathlib.Path(args.action_path)
    if not action_path.exists():
        raise FileNotFoundError(f"Action file not found: {action_path}")

    actions = np.asarray(np.load(action_path), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions with shape [T, 7], got {actions.shape}")

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not (0 <= args.task_id < task_suite.n_tasks):
        raise ValueError(f"task_id must be in [0, {task_suite.n_tasks - 1}], got {args.task_id}")

    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if not (0 <= args.initial_state_idx < len(initial_states)):
        raise ValueError(
            f"initial_state_idx must be in [0, {len(initial_states) - 1}], got {args.initial_state_idx}"
        )

    env, task_description = _get_libero_env(
        task=task,
        resolution=LIBERO_ENV_RESOLUTION,
        seed=args.seed,
        enable_depth=True,
    )

    aligner = DepthGuidedAligner(DepthGuidedAlignConfig(enabled=True))
    aligner.reset_episode()

    wrist_cam_id = _get_camera_id(env, "robot0_eye_in_hand")
    wrist_cam_fovy_deg = _get_camera_fovy_deg(env, wrist_cam_id)

    env.reset()
    obs = env.set_init_state(initial_states[args.initial_state_idx])

    out_dir = pathlib.Path(args.trace_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = action_path.stem
    trace_dir = out_dir / f"{args.task_suite_name}_task{args.task_id}_{stem}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = trace_dir / "frames"
    if args.save_align_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    trace_rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    action_i = 0
    triggered = False
    triggered_t: Optional[int] = None
    triggered_action_i: Optional[int] = None
    align_success = False
    align_success_t: Optional[int] = None
    gripper_close_started = False
    gripper_close_started_t: Optional[int] = None
    gripper_closed = False
    gripper_closed_t: Optional[int] = None
    done = False
    saved_overlay_frames = 0

    for t in range(args.max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            if done:
                break
            continue

        wrist_depth = _get_depth_observation(obs, "robot0_eye_in_hand_depth")
        wrist_img = _get_depth_observation(obs, "robot0_eye_in_hand_image")
        if wrist_img is not None:
            wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1]).astype(np.uint8)
        if wrist_depth is not None:
            wrist_depth = np.ascontiguousarray(wrist_depth[::-1, ::-1]).astype(np.float32)

        eef_quat = _get_obs_vector(obs, "robot0_eef_quat")
        gripper_qpos = _get_obs_vector(obs, "robot0_gripper_qpos")
        wrist_cam_rot_base = _get_camera_rotmat(env, wrist_cam_id)

        analysis = aligner.analyze_depth_frame(wrist_depth, include_mask=args.save_align_frames)
        override_action, align_event = aligner.get_control_action(
            wrist_depth,
            eef_quat=eef_quat,
            camera_rot_base=wrist_cam_rot_base,
            camera_fovy_deg=wrist_cam_fovy_deg,
        )
        align_mode = aligner.get_mode()
        override_gripper_cmd = None if override_action is None else float(override_action[6])

        if args.save_align_frames:
            should_save_frame = True
            if args.save_only_active_align_frames:
                should_save_frame = bool(align_event) or align_mode in ("align", "close") or bool(override_action is not None)
            if should_save_frame:
                source_shape = wrist_depth.shape[:2] if wrist_depth is not None else wrist_img.shape[:2] if wrist_img is not None else (1, 1)
                overlay = _render_depth_align_overlay(
                    wrist_img=wrist_img,
                    analysis=analysis,
                    source_shape=source_shape,
                    align_mode=align_mode,
                    align_event=align_event,
                )
                imageio.imsave(frames_dir / f"frame_{t:04d}.png", overlay)
                saved_overlay_frames += 1

        trace_rows.append(
            {
                "t": int(t),
                "action_i": int(action_i),
                "mode": align_mode,
                "align_success": int(align_success),
                "gripper_close_started": int(gripper_close_started),
                "gripper_closed": int(gripper_closed),
                "override_gripper_cmd": override_gripper_cmd,
                "obs_gripper_qpos": None if gripper_qpos is None else float(gripper_qpos[0]),
                "target_found": int(analysis.target_found),
                "should_trigger": int(analysis.should_trigger),
                "is_aligned_now": int(analysis.is_aligned),
                "x_error": analysis.x_error,
                "y_error": analysis.y_error,
                "center_dist": analysis.center_dist,
                "target_depth": analysis.target_depth,
                "depth_error": analysis.depth_error,
                "depth_gap_abs": analysis.depth_gap_abs,
                "event": align_event,
                "override_action": int(override_action is not None),
            }
        )

        if align_event:
            events.append({"t": int(t), "action_i": int(action_i), "event": align_event, "mode": align_mode})
            if "depth align done, close gripper" in align_event:
                align_success = True
                if align_success_t is None:
                    align_success_t = int(t)
                gripper_close_started = True
                if gripper_close_started_t is None:
                    gripper_close_started_t = int(t)
            if "depth align finished, back to VLA" in align_event:
                gripper_closed = True
                if gripper_closed_t is None:
                    gripper_closed_t = int(t)

        # Keep close state explicit even if event text is not emitted on every close frame.
        if align_mode == "close" and not gripper_close_started:
            gripper_close_started = True
            if gripper_close_started_t is None:
                gripper_close_started_t = int(t)

        if align_event and "triggered" in align_event:
            triggered = True
            triggered_t = int(t)
            triggered_action_i = int(action_i)
            if args.stop_before_align:
                break

        if action_i >= len(actions):
            break

        # Replay the saved policy action only (no model inference).
        obs, _, done, _ = env.step(actions[action_i].tolist())
        action_i += 1
        if done:
            break

    summary = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": task_description,
        "initial_state_idx": args.initial_state_idx,
        "action_path": str(action_path),
        "actions_total": int(len(actions)),
        "actions_executed": int(action_i),
        "triggered": bool(triggered),
        "triggered_t": triggered_t,
        "triggered_action_i": triggered_action_i,
        "align_success": bool(align_success),
        "align_success_t": align_success_t,
        "gripper_close_started": bool(gripper_close_started),
        "gripper_close_started_t": gripper_close_started_t,
        "gripper_closed": bool(gripper_closed),
        "gripper_closed_t": gripper_closed_t,
        "stop_before_align": bool(args.stop_before_align),
        "episode_done": bool(done),
        "saved_overlay_frames": int(saved_overlay_frames),
        "events": events,
    }

    if args.save_trace:
        csv_path = trace_dir / "frame_metrics.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "t",
                    "action_i",
                    "mode",
                    "align_success",
                    "gripper_close_started",
                    "gripper_closed",
                    "override_gripper_cmd",
                    "obs_gripper_qpos",
                    "target_found",
                    "should_trigger",
                    "is_aligned_now",
                    "x_error",
                    "y_error",
                    "center_dist",
                    "target_depth",
                    "depth_error",
                    "depth_gap_abs",
                    "event",
                    "override_action",
                ],
            )
            writer.writeheader()
            for row in trace_rows:
                writer.writerow(row)

        summary["csv_path"] = str(csv_path)
        if args.save_align_frames:
            summary["frames_dir"] = str(frames_dir)
        with (trace_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def _get_libero_env(task, resolution, seed, enable_depth=False):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": enable_depth,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _get_obs_vector(obs: dict, key: str) -> Optional[np.ndarray]:
    value = obs.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _get_camera_id(env: OffScreenRenderEnv, camera_name: str) -> Optional[int]:
    try:
        return int(env.sim.model.camera_name2id(camera_name))
    except Exception:
        return None


def _get_camera_fovy_deg(env: OffScreenRenderEnv, camera_id: Optional[int]) -> Optional[float]:
    if camera_id is None:
        return None
    try:
        fovy = float(env.sim.model.cam_fovy[camera_id])
    except Exception:
        return None
    if not np.isfinite(fovy):
        return None
    return fovy


def _get_camera_rotmat(env: OffScreenRenderEnv, camera_id: Optional[int]) -> Optional[np.ndarray]:
    if camera_id is None:
        return None
    try:
        cam_xmat = np.asarray(env.sim.data.cam_xmat[camera_id], dtype=np.float64)
    except Exception:
        return None
    if cam_xmat.size != 9:
        return None
    return cam_xmat.reshape(3, 3)


def _get_depth_observation(obs: dict, key: str) -> Optional[np.ndarray]:
    value = obs.get(key)
    if value is None:
        return None
    return np.asarray(value)


def _resize_mask_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if mask.shape == (out_h, out_w):
        return mask
    y_idx = np.clip(np.round(np.linspace(0, mask.shape[0] - 1, out_h)).astype(np.int32), 0, mask.shape[0] - 1)
    x_idx = np.clip(np.round(np.linspace(0, mask.shape[1] - 1, out_w)).astype(np.int32), 0, mask.shape[1] - 1)
    return mask[np.ix_(y_idx, x_idx)]


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    color_arr = np.asarray(color, dtype=np.float32)
    blended = image[mask].astype(np.float32) * (1.0 - alpha) + color_arr * alpha
    image[mask] = blended.astype(np.uint8)


def _draw_cross(image: np.ndarray, x: float, y: float, color: Tuple[int, int, int], size: int = 4) -> None:
    h, w = image.shape[:2]
    cx = int(round(x))
    cy = int(round(y))
    if not (0 <= cx < w and 0 <= cy < h):
        return
    for dx in range(-size, size + 1):
        xx = cx + dx
        if 0 <= xx < w:
            image[cy, xx] = color
    for dy in range(-size, size + 1):
        yy = cy + dy
        if 0 <= yy < h:
            image[yy, cx] = color


def _render_depth_align_overlay(
    wrist_img: Optional[np.ndarray],
    analysis: DepthFrameAnalysis,
    source_shape: Tuple[int, int],
    align_mode: Optional[str],
    align_event: Optional[str],
) -> np.ndarray:
    if wrist_img is None:
        out_h, out_w = source_shape
        overlay = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    else:
        overlay = np.asarray(wrist_img, dtype=np.uint8).copy()
        if overlay.ndim == 2:
            overlay = np.repeat(overlay[..., None], 3, axis=-1)

    out_h, out_w = overlay.shape[:2]
    src_h, src_w = source_shape

    if analysis.gripper_mask is not None:
        gripper_mask = _resize_mask_nearest(analysis.gripper_mask, out_h, out_w)
        _blend_mask(overlay, gripper_mask, color=(0, 0, 255), alpha=0.25)

    if analysis.near_mask is not None:
        near_mask = _resize_mask_nearest(analysis.near_mask, out_h, out_w)
        _blend_mask(overlay, near_mask, color=(255, 0, 0), alpha=0.35)

    sx = out_w / max(float(src_w), 1.0)
    sy = out_h / max(float(src_h), 1.0)

    anchor_x = analysis.target_anchor[0] * sx
    anchor_y = analysis.target_anchor[1] * sy
    _draw_cross(overlay, anchor_x, anchor_y, color=(0, 255, 255), size=5)

    if analysis.target_cx is not None and analysis.target_cy is not None:
        target_x = analysis.target_cx * sx
        target_y = analysis.target_cy * sy
        _draw_cross(overlay, target_x, target_y, color=(255, 0, 0), size=4)

    if align_event:
        if "triggered" in align_event:
            event_color = (255, 255, 0)
        elif "done" in align_event:
            event_color = (0, 255, 0)
        elif "timed out" in align_event:
            event_color = (255, 0, 255)
        else:
            event_color = (255, 255, 255)
        overlay[0:10, 0:40] = event_color
    elif align_mode == "align":
        overlay[0:10, 0:40] = (255, 165, 0)
    elif align_mode == "close":
        overlay[0:10, 0:40] = (0, 255, 0)

    return overlay


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
