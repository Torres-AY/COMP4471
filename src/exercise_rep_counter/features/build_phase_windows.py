"""CLI entry point for building annotated Kaggle phase windows."""

import argparse
from pathlib import Path

from .phase_windows import build_phase_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/kaggle"))
    parser.add_argument("--window-length", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    datasets, normalizer = build_phase_windows(
        args.manifest, args.annotations, args.window_length, args.stride
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    normalizer.save(args.output_root / "phase_normalization.npz")
    for split, dataset in datasets.items():
        dataset.save(args.output_root / f"phase_{split}_w{args.window_length}.npz")


if __name__ == "__main__":
    main()
