import collections
import csv
import dataclasses
import json
import logging
import math
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from augmented.depth_guided_align import DepthFrameAnalysis, DepthGuidedAlignConfig, DepthGuidedAligner
from accelerate.action_quant import action_quant
from accelerate.skip_vla import fit_next_action_chunk, should_skip_vla
from accelerate.action_stage_determine import stage_determine

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_id: Optional[int] = None  # Specific task ID to evaluate
    fixed_initial_state_idx: Optional[int] = None  # Use this init state index for every trial when set
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_video: bool = False  # Whether to save videos
    save_failure: bool = False  # Whether to save failed episode data (actions + frames) for analysis
    save_frame: bool = False  # Whether to save per-episode frame images
    save_depth: bool = False  # Whether to save per-episode depth maps (raw + visualization)
    save_actions: bool = False  # Whether to save per-episode actions
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_path: str = "data/libero/eval_results/vanilla.json"  # Path to save final results
    frame_out_path: str = "data/libero/frame"  # Path to save per-episode frame images
    depth_out_path: str = "data/libero/depth"  # Path to save per-episode depth maps
    actions_out_path: str = "data/libero/actions"  # Path to save per-episode actions
    failure_path: str = "data/libero/failure"  # Path to save failed episode data for analysis

    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # Acceleration parameters
    #################################################################################################################
    action_quant: bool = False  # Whether to use action quantization
    action_quant_method: str = "fixed"  # "fixed" or "adaptive"
    action_quant_steps: int = 2  # Number of steps to aggregate for "fixed" method
    action_quant_threshold: float = 0.03  # Threshold for "adaptive" method

    vla_skip: bool = False  # Whether to enable chunk-level VLA skipping during evaluation
    z_xy_rate_skip: float = 0.4  # Skip if max_z / max_xy is below this threshold for the previous chunk
    z_max_skip: float = 0.3  # Skip if max absolute z in the previous chunk is below this threshold

    action_aware_chunk: bool = False  # Whether to use action-aware chunking
    translation_threshold: float = 0.5  # Threshold for translation speed to determine coarse stage
    z_threshold: float = 0.1  # Threshold for z speed to determine
    fine_chunk_steps: int = 10  # Chunk size used for fine stage when action-aware chunking is enabled
    coarse_chunk_steps: int = 15  # Chunk size used for coarse stage when action-aware chunking is enabled

    #################################################################################################################
    # Depth-guided fine alignment parameters
    #################################################################################################################
    enable_depth_align: bool = False  # Use depth heuristic to align before gripper close

    save_depth_align_trace: bool = False  # Save per-frame detection overlays and alignment state/events during eval
    depth_align_trace_out_path: str = "data/libero/depth_align_trace"  # Output directory for depth-align traces

def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    depth_align_config = _build_depth_align_config(args)
    depth_aligner = DepthGuidedAligner(depth_align_config) if depth_align_config.enabled else None

    # Start evaluation
    total_episodes, total_successes = 0, 0
    success_episode_steps = []
    infer_counts_per_success = []

    if args.task_id is None:
        task_ids = range(num_tasks_in_suite)
    else:
        if not 0 <= args.task_id < num_tasks_in_suite:
            raise ValueError(f"task_id must be in [0, {num_tasks_in_suite - 1}], got {args.task_id}")
        task_ids = [args.task_id]
        logging.info(f"Evaluating only task_id={args.task_id} in suite {args.task_suite_name}")

    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        if args.fixed_initial_state_idx is not None:
            if not 0 <= args.fixed_initial_state_idx < len(initial_states):
                raise ValueError(
                    f"fixed_initial_state_idx must be in [0, {len(initial_states) - 1}], got {args.fixed_initial_state_idx}"
                )
        elif args.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"num_trials_per_task={args.num_trials_per_task} exceeds available initial states ({len(initial_states)}). "
                "Set a smaller num_trials_per_task or set fixed_initial_state_idx."
            )

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
            enable_depth=args.save_depth or args.enable_depth_align,
        )
        wrist_cam_id = _get_camera_id(env, "robot0_eye_in_hand")
        wrist_cam_fovy_deg = _get_camera_fovy_deg(env, wrist_cam_id)
        if depth_aligner is not None and wrist_cam_id is None:
            logging.warning("DepthAlign: wrist camera id not found, pose-guided XY will fall back to gain mapping")

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            current_chunk_actions = []
            prev_executed_chunk = None
            if depth_aligner is not None:
                depth_aligner.reset_episode()
                _gripper_mask_initialized = False

            # Set initial states
            init_state_idx = args.fixed_initial_state_idx if args.fixed_initial_state_idx is not None else episode_idx
            obs = env.set_init_state(initial_states[init_state_idx])

            # Setup
            t = 0
            done = False
            last_gripper_cmd = float(LIBERO_DUMMY_ACTION[6])
            replay_images = []
            replay_wrist_images = []
            replay_depths = []
            replay_wrist_depths = []
            episode_actions = []
            episode_infer_count = 0
            episode_global_idx = total_episodes + 1

            frame_episode_dir = None
            frame_agent_dir = None
            frame_wrist_dir = None
            if args.save_frame:
                frame_episode_dir = pathlib.Path(args.frame_out_path) / f"episode_{episode_global_idx:03d}"
                frame_episode_dir.mkdir(parents=True, exist_ok=True)
                frame_agent_dir = frame_episode_dir / "frames"
                frame_wrist_dir = frame_episode_dir / "wrist_frames"
                frame_agent_dir.mkdir(parents=True, exist_ok=True)
                frame_wrist_dir.mkdir(parents=True, exist_ok=True)

            depth_episode_dir = None
            depth_agent_vis_dir = None
            depth_wrist_vis_dir = None
            if args.save_depth:
                depth_episode_dir = pathlib.Path(args.depth_out_path) / f"episode_{episode_global_idx:03d}"
                depth_episode_dir.mkdir(parents=True, exist_ok=True)
                depth_agent_vis_dir = depth_episode_dir / "depth_vis"
                depth_wrist_vis_dir = depth_episode_dir / "wrist_depth_vis"
                depth_agent_vis_dir.mkdir(parents=True, exist_ok=True)
                depth_wrist_vis_dir.mkdir(parents=True, exist_ok=True)

            depth_align_trace_episode_dir = None
            depth_align_trace_frames_dir = None
            depth_align_trace_rows = []
            depth_align_events = []
            if args.save_depth_align_trace and depth_aligner is not None:
                depth_align_trace_episode_dir = (
                    pathlib.Path(args.depth_align_trace_out_path) / f"episode_{episode_global_idx:03d}"
                )
                depth_align_trace_frames_dir = depth_align_trace_episode_dir / "frames"
                depth_align_trace_frames_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        last_gripper_cmd = float(LIBERO_DUMMY_ACTION[6])
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    depth = _get_depth_observation(obs, "agentview_depth")
                    wrist_depth = _get_depth_observation(obs, "robot0_eye_in_hand_depth")
                    if depth is not None:
                        depth = np.ascontiguousarray(depth[::-1, ::-1]).astype(np.float32)
                    if wrist_depth is not None:
                        wrist_depth = np.ascontiguousarray(wrist_depth[::-1, ::-1]).astype(np.float32)

                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    replay_wrist_images.append(wrist_img)
                    if depth is not None:
                        replay_depths.append(depth)
                    if wrist_depth is not None:
                        replay_wrist_depths.append(wrist_depth)

                    if args.save_frame and frame_agent_dir is not None and frame_wrist_dir is not None:
                        imageio.imsave(frame_agent_dir / f"frame_{t:03d}.png", img)
                        imageio.imsave(frame_wrist_dir / f"wrist_frame_{t:03d}.png", wrist_img)

                    if args.save_depth and depth_agent_vis_dir is not None and depth_wrist_vis_dir is not None:
                        if depth is not None:
                            imageio.imsave(depth_agent_vis_dir / f"depth_{t:03d}.png", _depth_to_vis(depth))
                        if wrist_depth is not None:
                            imageio.imsave(depth_wrist_vis_dir / f"wrist_depth_{t:03d}.png", _depth_to_vis(wrist_depth))

                    align_analysis = None
                    align_event = None
                    align_mode = None
                    override_action = None
                    if depth_aligner is not None:
                        align_depth = wrist_depth
                        eef_pos = _get_obs_vector(obs, "robot0_eef_pos")
                        eef_quat = _get_obs_vector(obs, "robot0_eef_quat")
                        wrist_cam_rot_base = _get_camera_rotmat(env, wrist_cam_id)
                        if not _gripper_mask_initialized and align_depth is not None:
                            depth_aligner.initialize_gripper_mask(align_depth)
                            _gripper_mask_initialized = True
                        align_analysis = depth_aligner.analyze_depth_frame(
                            align_depth,
                            include_mask=args.save_depth_align_trace,
                            gripper_cmd=last_gripper_cmd,
                        )
                        override_action, align_event = depth_aligner.get_control_action(
                            align_depth,
                            eef_pos=eef_pos,
                            eef_quat=eef_quat,
                            camera_rot_base=wrist_cam_rot_base,
                            camera_fovy_deg=wrist_cam_fovy_deg,
                            gripper_cmd=last_gripper_cmd,
                        )
                        align_mode = depth_aligner.get_mode()
                        if align_event:
                            logging.info(
                                "[DepthAlign] %s (t=%d)",
                                align_event,
                                t,
                            )
                            if args.save_depth_align_trace:
                                depth_align_events.append(
                                    {
                                        "t": int(t),
                                        "event": align_event,
                                        "mode": align_mode,
                                    }
                                )

                        if args.save_depth_align_trace and depth_align_trace_frames_dir is not None and align_analysis is not None:
                            source_shape = align_depth.shape[:2] if align_depth is not None else wrist_img.shape[:2]
                            overlay_img = _render_depth_align_overlay(
                                wrist_img,
                                align_analysis,
                                source_shape=source_shape,
                                align_mode=align_mode,
                                align_event=align_event,
                            )
                            imageio.imsave(depth_align_trace_frames_dir / f"frame_{t:03d}.png", overlay_img)

                            is_aligned_now = bool(align_analysis.is_aligned)
                            depth_align_trace_rows.append(
                                {
                                    "t": int(t),
                                    "mode": align_mode,
                                    "override_action": int(override_action is not None),
                                    "gripper_cmd": align_analysis.gripper_cmd,
                                    "can_run_detection_now": int(align_analysis.can_run_detection_now),
                                    "target_found": int(align_analysis.target_found),
                                    "should_trigger": int(align_analysis.should_trigger),
                                    "is_aligned_now": int(is_aligned_now),
                                    "target_x": align_analysis.target_cx,
                                    "target_y": align_analysis.target_cy,
                                    "x_error": align_analysis.x_error,
                                    "y_error": align_analysis.y_error,
                                    "anchor_x": align_analysis.target_anchor[0],
                                    "anchor_y": align_analysis.target_anchor[1],
                                    "center_dist": align_analysis.center_dist,
                                    "target_depth": align_analysis.target_depth,
                                    "depth_error": align_analysis.depth_error,
                                    "depth_gap_abs": align_analysis.depth_gap_abs,
                                    "pixel_count": align_analysis.pixel_count,
                                    "event": align_event,
                                }
                            )

                        if override_action is not None:
                            # Override any pending plan while the depth-based controller is active.
                            action_plan.clear()
                            current_chunk_actions = []
                            prev_executed_chunk = None

                    if override_action is None and not action_plan:
                        # Finished executing previous action chunk -- either infer a new chunk
                        # or skip one VLA call by fitting next chunk from the previous executed chunk.
                        if args.vla_skip and should_skip_vla(
                            prev_executed_chunk,
                            z_xy_rate_skip=args.z_xy_rate_skip,
                            z_max_skip=args.z_max_skip,
                        ):
                            action_chunk = fit_next_action_chunk(prev_executed_chunk)
                        else:
                            stage = "fine"
                            desired_steps = args.replan_steps
                            if args.action_aware_chunk:
                                if prev_executed_chunk is not None and len(prev_executed_chunk) > 0:
                                    stage = stage_determine(
                                        prev_executed_chunk,
                                        args.translation_threshold,
                                        args.z_threshold,
                                    )
                                desired_steps = args.coarse_chunk_steps if stage == "coarse" else args.fine_chunk_steps

                            # Prepare observations dict
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                            }
                            # Only override model inference chunk size in action-aware mode.
                            # When disabled, keep original flow: model returns default chunk and we slice locally.
                            if args.action_aware_chunk:
                                element["__sample_kwargs"] = {"action_horizon": int(desired_steps)}

                            # Query model to get action
                            response = client.infer(element)
                            episode_infer_count += 1
                            action_chunk = np.asarray(response["actions"], dtype=np.float32)
                            
                            assert (
                                len(action_chunk) >= desired_steps
                            ), f"We want to replan {desired_steps} steps, but policy only predicts {len(action_chunk)} steps."

                            action_chunk = action_chunk[:desired_steps]

                        action_chunk = np.asarray(action_chunk, dtype=np.float32)

                        if args.action_quant:
                            action_chunk = action_quant(
                                action_chunk, 
                                args.action_quant_method,
                                args.action_quant_steps,
                                args.action_quant_threshold,
                            )

                        current_chunk_actions = []
                        action_plan.extend(action_chunk)

                    if override_action is not None:
                        action = np.asarray(override_action, dtype=np.float32)
                    else:
                        action = action_plan.popleft()
                        current_chunk_actions.append(np.asarray(action, dtype=np.float32))

                    episode_actions.append(action)
                    if action.shape[0] > 6 and np.isfinite(action[6]):
                        last_gripper_cmd = float(action[6])
                    else:
                        last_gripper_cmd = None

                    if not action_plan and current_chunk_actions:
                        prev_executed_chunk = np.asarray(current_chunk_actions, dtype=np.float32)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        success_episode_steps.append(t - args.num_steps_wait + 1)
                        infer_counts_per_success.append(episode_infer_count)
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            if args.save_depth_align_trace and depth_align_trace_episode_dir is not None:
                _save_depth_align_trace(
                    depth_align_trace_episode_dir,
                    depth_align_trace_rows,
                    depth_align_events,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    initial_state_idx=init_state_idx,
                )

            if args.save_actions:
                actions_dir = pathlib.Path(args.actions_out_path)
                actions_dir.mkdir(parents=True, exist_ok=True)
                action_save_path = actions_dir / f"episode_{total_episodes:03d}.npy"
                np.save(action_save_path, np.asarray(episode_actions, dtype=np.float32))
                logging.info(f"Saved actions to {action_save_path}")

            if args.save_depth and depth_episode_dir is not None:
                if replay_depths:
                    np.save(depth_episode_dir / "depth.npy", np.asarray(replay_depths, dtype=np.float32))
                if replay_wrist_depths:
                    np.save(depth_episode_dir / "wrist_depth.npy", np.asarray(replay_wrist_depths, dtype=np.float32))

            # Failure analysis
            if args.save_failure and not done:
                failed_path = pathlib.Path(args.failure_path) / f"episode_{total_episodes:03d}"
                failed_path.mkdir(parents=True, exist_ok=True)
                frames_dir = failed_path / "frames"
                wrist_frames_dir = failed_path / "wrist_frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                wrist_frames_dir.mkdir(parents=True, exist_ok=True)

                if args.save_depth:
                    depth_dir = failed_path / "depth"
                    wrist_depth_dir = failed_path / "wrist_depth"
                    depth_dir.mkdir(parents=True, exist_ok=True)
                    wrist_depth_dir.mkdir(parents=True, exist_ok=True)

                np.save(failed_path / "actions.npy", np.asarray(episode_actions, dtype=np.float32))

                for frame_idx, frame in enumerate(replay_images):
                    imageio.imsave(frames_dir / f"frame_{frame_idx:03d}.png", frame)

                for frame_idx, wrist_frame in enumerate(replay_wrist_images):
                    imageio.imsave(wrist_frames_dir / f"wrist_frame_{frame_idx:03d}.png", wrist_frame)

                if args.save_depth:
                    if replay_depths:
                        np.save(failed_path / "depth.npy", np.asarray(replay_depths, dtype=np.float32))
                        for frame_idx, depth_frame in enumerate(replay_depths):
                            imageio.imsave(depth_dir / f"depth_{frame_idx:03d}.png", _depth_to_vis(depth_frame))
                    if replay_wrist_depths:
                        np.save(failed_path / "wrist_depth.npy", np.asarray(replay_wrist_depths, dtype=np.float32))
                        for frame_idx, depth_frame in enumerate(replay_wrist_depths):
                            imageio.imsave(
                                wrist_depth_dir / f"wrist_depth_{frame_idx:03d}.png",
                                _depth_to_vis(depth_frame),
                            )

            # Save a replay video of the episode
            if args.save_video:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"steps taken: {t - args.num_steps_wait + 1}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            logging.info(f"# model inferences for this episode: {episode_infer_count}")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    avg_success_steps = float(np.mean(success_episode_steps)) if success_episode_steps else 0.0
    avg_infer_count = float(np.mean(infer_counts_per_success)) if infer_counts_per_success else 0.0

    logging.info(f"Total success rate: {final_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Average steps for successful episodes: {avg_success_steps:.2f}")
    logging.info(f"Average model inferences: {avg_infer_count:.2f}")
    if args.results_path:
        # Convert args to dict and filter out unwanted keys
        args_dict = dataclasses.asdict(args)
        keys_to_exclude = {
            "host",
            "port",
            "save_failure",
            "save_video",
            "video_out_path",
            "save_frame",
            "frame_out_path",
            "save_depth",
            "depth_out_path",
            "save_actions",
            "actions_out_path",
            "save_depth_align_trace",
            "depth_align_trace_out_path",
            "results_path",
            "failure_path",
        }
        filtered_args = {k: v for k, v in args_dict.items() if k not in keys_to_exclude}

        results = {
            **filtered_args,  # Merge filtered args into results
            "total_success_rate": final_success_rate,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "avg_success_steps": avg_success_steps,
            "avg_infer_count": avg_infer_count,
        }
        pathlib.Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to {args.results_path}")


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
    wrist_img: np.ndarray,
    analysis: DepthFrameAnalysis,
    source_shape: Tuple[int, int],
    align_mode: Optional[str],
    align_event: Optional[str],
) -> np.ndarray:
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


def _save_depth_align_trace(
    trace_episode_dir: pathlib.Path,
    frame_rows: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    task_id: int,
    episode_idx: int,
    initial_state_idx: int,
) -> None:
    trace_episode_dir.mkdir(parents=True, exist_ok=True)

    csv_path = trace_episode_dir / "frame_metrics.csv"
    fieldnames = [
        "t",
        "mode",
        "override_action",
        "gripper_cmd",
        "can_run_detection_now",
        "target_found",
        "should_trigger",
        "is_aligned_now",
        "target_x",
        "target_y",
        "x_error",
        "y_error",
        "anchor_x",
        "anchor_y",
        "center_dist",
        "target_depth",
        "depth_error",
        "depth_gap_abs",
        "pixel_count",
        "event",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_rows:
            writer.writerow(row)

    first_trigger_t = next((e["t"] for e in events if "triggered" in e.get("event", "")), None)
    first_aligned_t = next((e["t"] for e in events if "done" in e.get("event", "")), None)

    summary = {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "initial_state_idx": initial_state_idx,
        "frames_logged": len(frame_rows),
        "events_logged": len(events),
        "first_trigger_t": first_trigger_t,
        "first_aligned_t": first_aligned_t,
        "events": events,
        "csv_path": str(csv_path),
    }
    with (trace_episode_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def _get_libero_env(task, resolution, seed, enable_depth=False):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": enable_depth,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _build_depth_align_config(args: Args) -> DepthGuidedAlignConfig:
    return DepthGuidedAlignConfig(
        enabled=args.enable_depth_align,
    )


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


def _depth_to_vis(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 3:
        depth = depth[..., 0]

    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    valid = depth[finite]
    lo = np.percentile(valid, 1.0)
    hi = np.percentile(valid, 99.0)
    if hi <= lo:
        hi = lo + 1e-6

    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    gray = (normalized * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
