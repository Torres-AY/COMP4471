"""Build fixed-size windows with human-annotated phase labels."""

from pathlib import Path

import numpy as np

from exercise_rep_counter.annotation.phase_csv import labels_for_video, read_phase_intervals
from exercise_rep_counter.data.manifest import read_manifest
from .mediapipe import load_engineered_features
from .preprocessing import FeatureNormalizer
from .windows import WindowDataset, _windows


def build_phase_windows(
    manifest_path: str | Path,
    annotations_path: str | Path,
    window_length: int = 60,
    stride: int = 5,
) -> tuple[dict[str, WindowDataset], FeatureNormalizer]:
    """Build train/validation/test phase datasets from annotated videos."""

    records = read_manifest(manifest_path)
    intervals = read_phase_intervals(annotations_path)
    if any(not record.split for record in records):
        raise ValueError("Manifest records must have video-level train/validation/test splits")
    loaded = [
        (record, load_engineered_features(record.feature_path))
        for record in records
    ]
    normalizer = FeatureNormalizer.fit(
        [features for record, features in loaded if record.split == "train"]
    )
    split_items: dict[str, list[tuple]] = {"train": [], "validation": [], "test": []}
    for record, features in loaded:
        features = normalizer.transform(features)
        windows, centers = _windows(features, window_length, stride)
        phases, repetitions = labels_for_video(intervals, record.sample_id, len(features))
        valid = np.array(
            [phases[center] is not None for center in centers], dtype=bool
        )
        windows = windows[valid]
        centers = centers[valid]
        labels = np.array([phases[center] for center in centers], dtype="<U12")
        rep_ids = np.array([repetitions[center] or "" for center in centers], dtype="<U32")
        split_items[record.split].append((record, windows, centers, labels, rep_ids))

    datasets = {
        split: _combine_phase_windows(items, normalizer.mean.shape[0], window_length)
        for split, items in split_items.items()
    }
    return datasets, normalizer


def _combine_phase_windows(
    items: list[tuple], feature_dim: int, window_length: int
) -> WindowDataset:
    nonempty = [item for item in items if len(item[1])]
    if not nonempty:
        return WindowDataset.empty(window_length, feature_dim)
    return WindowDataset(
        features=np.concatenate([item[1] for item in nonempty]),
        labels=np.concatenate([item[3] for item in nonempty]),
        subject_ids=np.concatenate(
            [np.full(len(item[1]), item[0].subject_id) for item in nonempty]
        ),
        sample_ids=np.concatenate(
            [np.full(len(item[1]), item[0].sample_id) for item in nonempty]
        ),
        center_frames=np.concatenate([item[2] for item in nonempty]),
    )
