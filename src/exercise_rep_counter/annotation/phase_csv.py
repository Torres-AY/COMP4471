"""Read and validate frame-phase interval annotations."""

from dataclasses import dataclass
import csv
from pathlib import Path

from exercise_rep_counter.config import PHASE_LABELS


@dataclass(frozen=True)
class PhaseInterval:
    video_id: str
    start_frame: int
    end_frame: int
    phase: str
    rep_id: str


def read_phase_intervals(path: str | Path) -> list[PhaseInterval]:
    """Read inclusive frame intervals from a phase annotation CSV."""

    required = {"video_id", "start_frame", "end_frame", "phase", "rep_id"}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"Phase CSV columns must be exactly {sorted(required)}")
        intervals = []
        for row in reader:
            interval = PhaseInterval(
                video_id=row["video_id"],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                phase=row["phase"],
                rep_id=row["rep_id"],
            )
            if interval.start_frame < 0 or interval.end_frame < interval.start_frame:
                raise ValueError(f"Invalid frame interval: {interval}")
            if interval.phase not in PHASE_LABELS:
                raise ValueError(f"Unknown phase {interval.phase!r}")
            intervals.append(interval)
    _validate_no_overlaps(intervals)
    return intervals


def _validate_no_overlaps(intervals: list[PhaseInterval]) -> None:
    by_video: dict[str, list[PhaseInterval]] = {}
    for interval in intervals:
        by_video.setdefault(interval.video_id, []).append(interval)
    for video_id, video_intervals in by_video.items():
        ordered = sorted(video_intervals, key=lambda item: item.start_frame)
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_frame <= previous.end_frame:
                raise ValueError(f"Overlapping intervals for {video_id}")


def labels_for_video(
    intervals: list[PhaseInterval], video_id: str, num_frames: int
) -> tuple[list[str | None], list[str | None]]:
    """Expand intervals into phase and repetition labels for one video."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    phases: list[str | None] = [None] * num_frames
    repetitions: list[str | None] = [None] * num_frames
    for interval in intervals:
        if interval.video_id != video_id:
            continue
        if interval.end_frame >= num_frames:
            raise ValueError(f"Annotation exceeds video length for {video_id}")
        for frame in range(interval.start_frame, interval.end_frame + 1):
            phases[frame] = interval.phase
            repetitions[frame] = interval.rep_id
    return phases, repetitions
