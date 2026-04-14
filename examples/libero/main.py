import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

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
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_video: bool = False  # Whether to save videos
    save_failure: bool = False  # Whether to save failed episode data (actions + frames) for analysis
    save_frame: bool = False  # Whether to save per-episode frame images
    save_actions: bool = False  # Whether to save per-episode actions
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_path: str = "data/libero/eval_results/vanilla.json"  # Path to save final results
    frame_out_path: str = "data/libero/frame"  # Path to save per-episode frame images
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

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            current_chunk_actions = []
            prev_executed_chunk = None

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []
            replay_wrist_images = []
            episode_actions = []
            episode_infer_count = 0
            episode_global_idx = total_episodes + 1

            frame_episode_dir = None
            if args.save_frame:
                frame_episode_dir = pathlib.Path(args.frame_out_path) / f"episode_{episode_global_idx:05d}"
                frame_episode_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
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

                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    replay_wrist_images.append(wrist_img)

                    if args.save_frame and frame_episode_dir is not None:
                        imageio.imsave(frame_episode_dir / f"img_frame_{t:03d}.png", img)
                        imageio.imsave(frame_episode_dir / f"wrist_img_frame_{t:03d}.png", wrist_img)

                    if not action_plan:
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

                    action = action_plan.popleft()
                    current_chunk_actions.append(np.asarray(action, dtype=np.float32))
                    episode_actions.append(action)

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

            if args.save_actions:
                actions_dir = pathlib.Path(args.actions_out_path)
                actions_dir.mkdir(parents=True, exist_ok=True)
                action_save_path = actions_dir / f"episode_{total_episodes:05d}.npy"
                np.save(action_save_path, np.asarray(episode_actions, dtype=np.float32))
                logging.info(f"Saved actions to {action_save_path}")

            # Failure analysis
            if args.save_failure and not done:
                failed_path = pathlib.Path(args.failure_path) / f"episode_{total_episodes}"
                failed_path.mkdir(parents=True, exist_ok=True)
                frames_dir = failed_path / "frames"
                wrist_frames_dir = failed_path / "wrist_frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                wrist_frames_dir.mkdir(parents=True, exist_ok=True)

                np.save(failed_path / "actions.npy", np.asarray(episode_actions, dtype=np.float32))

                for frame_idx, frame in enumerate(replay_images):
                    imageio.imsave(frames_dir / f"frame_{frame_idx:03d}.png", frame)

                for frame_idx, wrist_frame in enumerate(replay_wrist_images):
                    imageio.imsave(wrist_frames_dir / f"wrist_frame_{frame_idx:03d}.png", wrist_frame)

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
            "save_actions",
            "actions_out_path",
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


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


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
