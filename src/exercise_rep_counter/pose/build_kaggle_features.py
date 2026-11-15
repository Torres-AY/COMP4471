"""Extract MediaPipe landmarks from all videos in the Kaggle manifest."""

import argparse
from pathlib import Path

from .extractor import extract_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/kaggle_manifest.csv"))
    parser.add_argument("--source-root", type=Path, default=Path("data/external/kaggle"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/kaggle"))
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/metadata/kaggle_processed_manifest.csv"),
    )
    args = parser.parse_args()
    extract_manifest(
        args.manifest,
        args.source_root,
        args.model,
        args.output_root,
        args.output_manifest,
    )


if __name__ == "__main__":
    main()
