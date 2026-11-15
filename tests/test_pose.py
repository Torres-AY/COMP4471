import numpy as np
import pytest

from exercise_rep_counter.pose.features import normalize_landmarks


def test_normalize_landmarks_centers_hips_and_scales_shoulders():
    landmarks = np.zeros((1, 33, 4), dtype=np.float32)
    landmarks[0, 23, :3] = [1, 1, 0]
    landmarks[0, 24, :3] = [3, 1, 0]
    landmarks[0, 11, :3] = [0, 0, 0]
    landmarks[0, 12, :3] = [4, 0, 0]
    landmarks[0, :, 3] = 1

    result = normalize_landmarks(landmarks)

    np.testing.assert_allclose(result[0, 23, :3], [-0.25, 0, 0])
    np.testing.assert_allclose(result[0, 24, :3], [0.25, 0, 0])
    np.testing.assert_allclose(result[0, 11, :3], [-0.5, -0.25, 0])


def test_normalize_landmarks_marks_invalid_scale_missing():
    landmarks = np.ones((2, 33, 4), dtype=np.float32)
    landmarks[0, 11, :3] = landmarks[0, 12, :3]
    landmarks[1, 11, 0] = 0

    result = normalize_landmarks(landmarks)

    assert np.all(result[0, :, 3] == 0)
    assert np.all(result[1, :, 3] == 1)


def test_normalize_landmarks_requires_media_pipe_shape():
    with pytest.raises(ValueError, match="33"):
        normalize_landmarks(np.zeros((2, 10, 4), dtype=np.float32))
