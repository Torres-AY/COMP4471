"""Feature preparation for MediaPipe landmark sequences."""

import numpy as np


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Center landmarks at the hip midpoint and scale by shoulder width.

    The input and output shape is ``[frames, 33, 4]`` with x/y/z/visibility.
    Visibility is preserved and is not geometrically transformed.
    """

    values = np.asarray(landmarks, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (33, 4):
        raise ValueError("Expected landmarks with shape [frames, 33, 4]")
    normalized = values.copy()
    left_hip, right_hip = 23, 24
    left_shoulder, right_shoulder = 11, 12
    center = (values[:, left_hip, :3] + values[:, right_hip, :3]) / 2
    scale = np.linalg.norm(
        values[:, left_shoulder, :3] - values[:, right_shoulder, :3], axis=1
    )
    valid = np.isfinite(values).all(axis=(1, 2)) & (scale > 1e-6)
    normalized[:, :, :3] = 0
    normalized[valid, :, :3] = (
        values[valid, :, :3] - center[valid, None, :]
    ) / scale[valid, None, None]
    normalized[~valid, :, 3] = 0
    return normalized
