
import numpy as np
from typing import Optional


def should_skip_vla(
    prev_executed_chunk: Optional[np.ndarray],
    z_xy_rate_skip: float,
    z_max_skip: float,
) -> bool:
    """Chunk-level skip rule: skip VLA if previous chunk is mostly planar and has small z motion."""
    if prev_executed_chunk is None or len(prev_executed_chunk) == 0:
        return False

    max_xy = np.max(np.maximum(np.abs(prev_executed_chunk[:, 0]), np.abs(prev_executed_chunk[:, 1])))
    max_xy = max(float(max_xy), 1e-4)
    max_z = float(np.max(np.abs(prev_executed_chunk[:, 2])))
    return (max_z / max_xy) < z_xy_rate_skip and max_z < z_max_skip


def fit_next_action_chunk(prev_executed_chunk: np.ndarray) -> np.ndarray:
    """Fit next chunk directly from previous chunk without maintaining an external sliding window."""
    prev = np.asarray(prev_executed_chunk, dtype=np.float32)
    n_steps, action_dim = prev.shape

    xyz = prev[:, :3]
    x = np.arange(n_steps, dtype=np.float32).reshape(-1, 1)
    x = np.hstack([x, np.ones_like(x)])
    y = xyz

    lambda_reg = 1e-4
    xtx = x.T @ x
    beta = np.linalg.solve(xtx + lambda_reg * np.eye(2, dtype=np.float32), x.T @ y)

    x_future = np.arange(n_steps, 2 * n_steps, dtype=np.float32).reshape(-1, 1)
    x_future = np.hstack([x_future, np.ones_like(x_future)])
    pred_xyz = x_future @ beta

    pred_xyz = np.clip(pred_xyz, -0.9, 0.9)

    if n_steps > 1:
        delta_max = np.max(np.abs(np.diff(y, axis=0)), axis=0)
    else:
        delta_max = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    last_xyz = y[-1]
    for i in range(n_steps):
        delta = pred_xyz[i] - last_xyz
        delta = np.clip(delta, -delta_max, delta_max)
        pred_xyz[i] = last_xyz + delta
        last_xyz = pred_xyz[i]

    pred_chunk = np.tile(prev[-1], (n_steps, 1))
    pred_chunk[:, :3] = pred_xyz

    # Keep the non-translation dimensions constant from the last executed action.
    if action_dim > 3:
        pred_chunk[:, 3:] = prev[-1, 3:]

    return pred_chunk.astype(np.float32)
