import numpy as np

def action_quant(
    action_chunk: np.ndarray,
    method: str = "fixed",         # "fixed" or "adaptive"
    quant_steps: int = 1,
    adaptive_threshold: float = 0.03,
) -> np.ndarray:
    """
    Quantize action chunks by aggregating steps.

    method="fixed": aggregate every `quant_steps` steps
    method="adaptive": aggregate based on significant waypoints using Corki

    Args:
        action_chunk: (T, 7) array
        quant_steps: int >= 1
        method: "fixed" | "adaptive"
        adaptive_threshold: waypoint threshold

    Returns:
        (T', 7) array
    """
    action_chunk = np.asarray(action_chunk, dtype=np.float32)

    if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
        raise ValueError(f"Expected action_chunk shape (T, 7), got {action_chunk.shape}")

    if method == "adaptive":
        quant_action_chunk = corki_merge(action_chunk, adaptive_threshold)
    elif method == "stage_aware":
        quant_action_chunk = stage_aware_merge(action_chunk, adaptive_threshold, quant_steps)
    elif method == "fixed":
        quant_action_chunk = fixed_step_merge(action_chunk, quant_steps)

    return quant_action_chunk

def fixed_step_merge(action_chunk: np.ndarray, quant_steps: int) -> np.ndarray:
    """
    Merge actions by fixed steps.
    For each segment of `quant_steps`, sum pose deltas and keep last gripper.
    """
    if quant_steps <= 1:
        return action_chunk

    n_steps = action_chunk.shape[0]

    # Full groups
    n_full_steps = (n_steps // quant_steps) * quant_steps
    if n_full_steps > 0:
        reshaped = action_chunk[:n_full_steps].reshape(-1, quant_steps, 7)
        pose_deltas = reshaped[..., :6].sum(axis=1)
        gripper = reshaped[..., -1, 6:]
        main_part = np.concatenate([pose_deltas, gripper], axis=-1)
    else:
        main_part = np.empty((0, 7), dtype=np.float32)

    # Remainder group
    if n_full_steps < n_steps:
        remainder = action_chunk[n_full_steps:]
        rem_pose = remainder[..., :6].sum(axis=0, keepdims=True)
        rem_gripper = remainder[-1:, 6:]
        rem_part = np.concatenate([rem_pose, rem_gripper], axis=-1)
        quant_action_chunk = np.concatenate([main_part, rem_part], axis=0)
    else:
        quant_action_chunk = main_part

    return quant_action_chunk

def get_waypoint_indices(action: np.ndarray, threshold: float) -> np.ndarray:
    """
    Corki-style important waypoint selection (all waypoints).
    action: (T, 7) delta action chunk
    return: sorted unique waypoint indices (excluding 0)
    """
    if action.shape[0] == 0:
        return np.array([], dtype=np.int64)

    traj = action.copy()
    traj[:, :6] = np.cumsum(traj[:, :6], axis=0)

    # ---------------------------------------------------------
    # PART 1: 基于几何距离的检测 (检测转折点)
    # 核心思想：看轨迹是否是一条直线。如果是直线，中间的点到首尾连线的距离应该很小。
    # 如果偏离太大，说明发生了转弯，那个转弯点就是 Waypoint。
    # ---------------------------------------------------------
    start_idx = 0
    local_max_A = []
    while start_idx + 2 < len(traj):
        p_st = traj[start_idx, :3]
        found = False
        for jth in range(start_idx + 2, len(traj)):
            p_ed = traj[jth, :3]
            distance_max = 0.0
            for kth in range(start_idx + 1, jth):
                p = traj[kth, :3]
                if np.degrees(np.arccos(np.dot(p_ed - p, p_ed - p_st) /
                                    (np.linalg.norm(p_ed - p) * np.linalg.norm(p_ed - p_st)))) > 90 or \
                    np.degrees(np.arccos(np.dot(p_st - p, p_st - p_ed) /
                                    (np.linalg.norm(p_st - p) * np.linalg.norm(p_st - p_ed)))) > 90:
                    distance = 10000
                else:
                    distance = np.sin(np.arccos(np.dot(p - p_st, p_ed - p_st) /
                                            (np.linalg.norm(p - p_st) * np.linalg.norm(p_ed - p_st)))) * np.linalg.norm(p - p_st)
                distance_max = max(distance_max, distance)
            if distance_max > threshold:
                local_max_A.append(jth - 1)
                start_idx = jth
                found = True
                break
        if not found:
            break
    
    # ---------------------------------------------------------
    # PART 2: 基于夹爪状态的检测
    # 核心思想：夹爪开合（抓取/释放）一定是关键动作，必须保留。
    # ---------------------------------------------------------
    def _gripper_state_changed(t: np.ndarray):
        t = np.vstack([t[:1], t])
        changed = np.sign(t[:-1, -1]) != np.sign(t[1:, -1])
        return np.where(changed)[0]

    gripper_changed = _gripper_state_changed(traj)
    one_before = gripper_changed[gripper_changed > 1] - 1
    last_frame = np.array([len(traj) - 1], dtype=np.int64)

    keyframe_inds = np.unique(np.concatenate([
        last_frame,
        gripper_changed,
        one_before,
        np.array(local_max_A, dtype=np.int64),
    ]))
    keyframe_inds.sort()
    keyframe_inds = keyframe_inds[keyframe_inds != 0]
    return keyframe_inds


def corki_merge(action_chunk: np.ndarray, threshold: float) -> np.ndarray:
    """
    Merge actions by all significant waypoints.
    For each segment [prev_wp, wp], sum pose deltas and keep last gripper.
    """

    waypoints = get_waypoint_indices(action_chunk, threshold)
    if waypoints.size == 0:
        return action_chunk

    merged = []
    start = 0
    for wp in waypoints:
        end = int(wp) + 1  # include wp
        segment = action_chunk[start:end]
        pose = segment[:, :6].sum(axis=0, keepdims=True)
        gripper = segment[-1:, 6:7]
        merged.append(np.concatenate([pose, gripper], axis=-1))
        start = end

    return np.concatenate(merged, axis=0)

def stage_aware_merge(action_chunk: np.ndarray, translation_threshold: float, quant_steps: int) -> np.ndarray:
    """
    Merge the entire action chunk if ALL actions have
    a translation speed strictly greater than the threshold.
    If ANY action's speed is <= threshold, return the original chunk unmerged.
    """
    if action_chunk.shape[0] == 0:
        return action_chunk
        
    # Calculate the translation speed (L2 norm of the first 3 dimensions)
    speeds = np.linalg.norm(action_chunk[:, :3], axis=1)
    z_speeds = action_chunk[:, 2]
    
    # If all speeds in the chunk are above the threshold, merge
    if np.all(speeds > translation_threshold) and (np.all(z_speeds > 0) or np.all(z_speeds < -0.3)):
        action_chunk = fixed_step_merge(action_chunk, quant_steps=quant_steps)
    
    # Otherwise, if even one step is not fast enough, do not merge at all
    return action_chunk