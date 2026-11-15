"""Convert sequential feature files into fixed-size training windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exercise_rep_counter.data.manifest import SampleRecord
from exercise_rep_counter.data.uiprmd_loader import load_sequence
from .preprocessing import FeatureNormalizer


@dataclass(frozen=True)
class WindowDataset:
    """Arrays required by a temporal classifier."""

    features: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    sample_ids: np.ndarray
    center_frames: np.ndarray

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            features=self.features,
            labels=self.labels,
            subject_ids=self.subject_ids,
            sample_ids=self.sample_ids,
            center_frames=self.center_frames,
        )

    @classmethod
    def empty(cls, window_length: int, feature_dim: int) -> "WindowDataset":
        return cls(
            features=np.empty((0, window_length, feature_dim), dtype=np.float32),
            labels=np.empty(0, dtype="<U1"),
            subject_ids=np.empty(0, dtype="<U1"),
            sample_ids=np.empty(0, dtype="<U1"),
            center_frames=np.empty(0, dtype=np.int64),
        )

    @classmethod
    def _from_items(
        cls,
        items: list[tuple[SampleRecord, np.ndarray, np.ndarray]],
        feature_dim: int,
        window_length: int,
    ) -> "WindowDataset":
        return _combine_windows(items, feature_dim, window_length)


def _windows(
    sequence: np.ndarray, window_length: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    if sequence.shape[0] < window_length:
        return np.empty((0, window_length, sequence.shape[1]), dtype=np.float32), np.empty(
            0, dtype=np.int64
        )
    starts = range(0, sequence.shape[0] - window_length + 1, stride)
    centers = np.fromiter(
        (start + window_length // 2 for start in starts), dtype=np.int64
    )
    windows = np.stack(
        [sequence[start : start + window_length] for start in range(
            0, sequence.shape[0] - window_length + 1, stride
        )]
    ).astype(np.float32)
    return windows, centers


def build_window_dataset(
    records: list[SampleRecord],
    source_root: str | Path,
    window_length: int,
    stride: int = 5,
    normalizer: FeatureNormalizer | None = None,
) -> tuple[dict[str, WindowDataset], FeatureNormalizer]:
    """Build split datasets, fitting normalization on training records only."""

    if not records:
        raise ValueError("Cannot build windows from an empty manifest")
    if any(not record.split for record in records):
        raise ValueError("Every record must have a train, validation, or test split")
    root = Path(source_root)
    loaded = [(record, load_sequence(root / record.source_path)) for record in records]
    if normalizer is None:
        normalizer = FeatureNormalizer.fit(
            [sequence for record, sequence in loaded if record.split == "train"]
        )

    split_arrays: dict[str, list[tuple[SampleRecord, np.ndarray, np.ndarray]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record, sequence in loaded:
        transformed = normalizer.transform(sequence)
        windows, centers = _windows(transformed, window_length, stride)
        split_arrays[record.split].append((record, windows, centers))

    datasets = {
        split: _combine_windows(items, normalizer.mean.shape[0], window_length)
        for split, items in split_arrays.items()
    }
    return datasets, normalizer


def _combine_windows(
    items: list[tuple[SampleRecord, np.ndarray, np.ndarray]],
    feature_dim: int,
    window_length: int,
) -> WindowDataset:
    feature_parts = [item[1] for item in items if len(item[1])]
    if feature_parts:
        features = np.concatenate(feature_parts)
        labels = np.concatenate(
            [np.full(len(item[1]), item[0].exercise) for item in items if len(item[1])]
        )
        subjects = np.concatenate(
            [np.full(len(item[1]), item[0].subject_id) for item in items if len(item[1])]
        )
        sample_ids = np.concatenate(
            [np.full(len(item[1]), item[0].sample_id) for item in items if len(item[1])]
        )
        centers = np.concatenate([item[2] for item in items if len(item[1])])
    else:
        features = np.empty((0, window_length, feature_dim), dtype=np.float32)
        labels = subjects = sample_ids = np.empty(0, dtype="<U1")
        centers = np.empty(0, dtype=np.int64)
    return WindowDataset(features, labels, subjects, sample_ids, centers)
