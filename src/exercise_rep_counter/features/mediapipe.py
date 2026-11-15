"""Seven engineered features derived from normalized MediaPipe landmarks."""

import numpy as np


_LEFT = {"shoulder": 11, "elbow": 13, "hip": 23, "knee": 25, "ankle": 27}
_RIGHT = {"shoulder": 12, "elbow": 14, "hip": 24, "knee": 26, "ankle": 28}


def _angle(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> np.ndarray:
    first_vector = first - vertex
    last_vector = last - vertex
    first_norm = np.linalg.norm(first_vector, axis=-1)
    last_norm = np.linalg.norm(last_vector, axis=-1)
    denominator = first_norm * last_norm
    cosine = np.divide(
        np.sum(first_vector * last_vector, axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-8,
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _side_features(landmarks: np.ndarray, side: dict[str, int]) -> np.ndarray:
    points = landmarks[:, :, :3]
    elbow = _angle(
        points[:, side["shoulder"]], points[:, side["elbow"]], points[:, side["wrist"]]
    )
    shoulder = _angle(
        points[:, side["elbow"]], points[:, side["shoulder"]], points[:, side["hip"]]
    )
    hip = _angle(
        points[:, side["shoulder"]], points[:, side["hip"]], points[:, side["knee"]]
    )
    knee = _angle(
        points[:, side["hip"]], points[:, side["knee"]], points[:, side["ankle"]]
    )
    return np.stack((elbow, shoulder, hip, knee), axis=1)


def engineered_features(landmarks: np.ndarray) -> np.ndarray:
    """Return ``[frames, 7]`` elbow/pose/motion features.

    Angles are in radians. The final two channels are frame-aligned backward
    differences, with the first frame set to zero.
    """

    values = np.asarray(landmarks, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (33, 4):
        raise ValueError("Expected normalized landmarks with shape [frames, 33, 4]")
    if not np.isfinite(values).all():
        raise ValueError("Landmarks contain NaN or infinite values")

    left = _side_features(values, {**_LEFT, "wrist": 15})
    right = _side_features(values, {**_RIGHT, "wrist": 16})
    angles = (left + right) / 2
    centroid_height = values[:, [23, 24], 1].mean(axis=1)
    centroid_velocity = np.diff(centroid_height, prepend=centroid_height[:1])
    elbow_velocity = np.diff(angles[:, 0], prepend=angles[:1, 0])
    result = np.column_stack((angles, centroid_height, centroid_velocity, elbow_velocity))
    return result.astype(np.float32)


def load_engineered_features(path: str) -> np.ndarray:
    """Load normalized landmarks from an extractor archive and convert them."""

    with np.load(path) as archive:
        if "normalized_landmarks" not in archive:
            raise ValueError(f"{path} does not contain normalized_landmarks")
        return engineered_features(archive["normalized_landmarks"])
