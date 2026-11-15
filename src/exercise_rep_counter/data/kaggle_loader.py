"""Discovery of video files from the Kaggle exercise dataset."""

from pathlib import Path
import re

import cv2

from .manifest import SampleRecord


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
EXERCISE_ALIASES = {
    "push-up": "pushup",
    "push_ups": "pushup",
    "pushups": "pushup",
    "pushup": "pushup",
    "sit-up": "situp",
    "sit_ups": "situp",
    "situps": "situp",
    "squat": "squat",
    "bicep_curl": "bicep_curl",
    "bicep-curl": "bicep_curl",
}


def _normalise_exercise(path: Path) -> str:
    normalized_path = re.sub(r"[_\-\s]+", " ", str(path).lower())
    for alias, normalized in EXERCISE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias.replace('_', ' '))}\b", normalized_path):
            return normalized
    tokens = re.split(r"[/\\_\-\s]+", str(path).lower())
    for token in tokens:
        if token in EXERCISE_ALIASES:
            return EXERCISE_ALIASES[token]
    return "unknown"


def discover_videos(root: str | Path) -> list[SampleRecord]:
    """Discover Kaggle videos and infer exercise labels from their paths.

    Unknown labels are retained so the manifest exposes layout problems instead
    of silently dropping samples.
    """

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Kaggle dataset directory does not exist: {root_path}")

    records: list[SampleRecord] = []
    for video_path in sorted(
        path for path in root_path.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS
    ):
        relative = video_path.relative_to(root_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        records.append(
            SampleRecord(
                sample_id=f"kaggle_{video_path.stem}",
                dataset="kaggle",
                subject_id="unknown",
                exercise=_normalise_exercise(relative),
                source_path=str(relative),
                num_frames=num_frames,
                fps=fps,
                label_type="exercise",
            )
        )
    if not records:
        raise ValueError(f"No supported video files found under {root_path}")
    return records
