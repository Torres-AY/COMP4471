"""Convert extracted Kaggle landmarks into normalized temporal windows."""

import argparse
from pathlib import Path

from exercise_rep_counter.data.manifest import read_manifest, write_manifest
from exercise_rep_counter.data.video_splits import split_by_record
from .mediapipe import load_engineered_features
from .preprocessing import FeatureNormalizer
from .windows import WindowDataset, _windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/kaggle_processed_manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/kaggle"))
    parser.add_argument("--window-length", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    if any(not record.split for record in records):
        records = split_by_record(records)
        write_manifest(records, args.manifest)
    feature_records = [
        (record, load_engineered_features(record.feature_path))
        for record in records
    ]
    normalizer = FeatureNormalizer.fit(
        [features for record, features in feature_records if record.split == "train"]
    )
    split_items = {"train": [], "validation": [], "test": []}
    for record, features in feature_records:
        windows, centers = _windows(
            normalizer.transform(features), args.window_length, args.stride
        )
        split_items[record.split].append((record, windows, centers))

    args.output_root.mkdir(parents=True, exist_ok=True)
    normalizer.save(args.output_root / "normalization.npz")
    for split, items in split_items.items():
        dataset = WindowDataset._from_items(
            items, normalizer.mean.shape[0], args.window_length
        )
        dataset.save(args.output_root / f"{split}_w{args.window_length}.npz")


if __name__ == "__main__":
    main()
