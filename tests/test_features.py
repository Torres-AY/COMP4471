import numpy as np
import pytest

from exercise_rep_counter.data.manifest import SampleRecord
from exercise_rep_counter.features.preprocessing import FeatureNormalizer
from exercise_rep_counter.features.mediapipe import engineered_features
from exercise_rep_counter.features.windows import build_window_dataset


def _record(tmp_path, sample_id, subject, split):
    source = tmp_path / f"{sample_id}.txt"
    np.savetxt(source, np.arange(20, dtype=np.float32).reshape(10, 2), delimiter=",")
    return SampleRecord(sample_id, "uiprmd", subject, "m01", source.name, split=split)


def test_normalizer_uses_training_sequences_only():
    normalizer = FeatureNormalizer.fit([np.zeros((2, 2), dtype=np.float32)])

    transformed = normalizer.transform(np.ones((1, 2), dtype=np.float32))

    np.testing.assert_allclose(transformed, np.ones((1, 2), dtype=np.float32))


def test_window_dataset_preserves_subject_split_and_centers(tmp_path):
    records = [
        _record(tmp_path, "train", "s01", "train"),
        _record(tmp_path, "validation", "s02", "validation"),
        _record(tmp_path, "test", "s03", "test"),
    ]

    datasets, _ = build_window_dataset(records, tmp_path, window_length=4, stride=2)

    assert datasets["train"].features.shape == (4, 4, 2)
    assert datasets["train"].center_frames.tolist() == [2, 4, 6, 8]
    assert set(datasets["test"].subject_ids) == {"s03"}


def test_window_dataset_rejects_missing_split(tmp_path):
    record = _record(tmp_path, "sample", "s01", "")

    with pytest.raises(ValueError, match="split"):
        build_window_dataset([record], tmp_path, window_length=4)


def test_engineered_features_have_seven_channels():
    landmarks = np.zeros((5, 33, 4), dtype=np.float32)
    landmarks[:, :, 3] = 1

    features = engineered_features(landmarks)

    assert features.shape == (5, 7)
    assert np.isfinite(features).all()
