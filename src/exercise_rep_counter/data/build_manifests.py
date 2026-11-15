"""Build source manifests from locally downloaded datasets."""

import argparse
from pathlib import Path

from .kaggle_loader import discover_videos
from .manifest import write_manifest
from .splits import split_by_subject
from .uiprmd_loader import discover_sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-root", type=Path)
    parser.add_argument("--uiprmd-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/metadata"))
    args = parser.parse_args()

    if args.kaggle_root is None and args.uiprmd_root is None:
        parser.error("provide at least one of --kaggle-root or --uiprmd-root")

    if args.kaggle_root is not None:
        write_manifest(
            discover_videos(args.kaggle_root),
            args.output_root / "kaggle_manifest.csv",
        )
    if args.uiprmd_root is not None:
        records = discover_sequences(args.uiprmd_root)
        write_manifest(
            split_by_subject(records),
            args.output_root / "uiprmd_manifest.csv",
        )


if __name__ == "__main__":
    main()
