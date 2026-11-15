"""Build normalized UI-PRMD training windows from a manifest."""

import argparse
from pathlib import Path

from exercise_rep_counter.data.manifest import read_manifest
from .windows import build_window_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/uiprmd_manifest.csv"))
    parser.add_argument("--source-root", type=Path, default=Path("data/external/uiprmd"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/uiprmd"))
    parser.add_argument("--window-length", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    datasets, normalizer = build_window_dataset(
        read_manifest(args.manifest),
        args.source_root,
        args.window_length,
        args.stride,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    normalizer.save(args.output_root / "normalization.npz")
    for split, dataset in datasets.items():
        dataset.save(args.output_root / f"{split}_w{args.window_length}.npz")


if __name__ == "__main__":
    main()
