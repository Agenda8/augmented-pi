import numpy as np

def stage_determine(action_chunk, translation_threshold=0.5, z_threshold=0.1):
    """
    Determine the action stage (coarse or fine) based on the action chunk.
    """
    speeds = np.linalg.norm(action_chunk[:, :3], axis=1)
    z_speeds = action_chunk[:, 2]
    rotation_speeds = np.linalg.norm(action_chunk[:, 3:6], axis=1)
    if np.all(speeds > translation_threshold) and np.all(abs(z_speeds) > z_threshold):
        return "coarse"
    else:
        return "fine"