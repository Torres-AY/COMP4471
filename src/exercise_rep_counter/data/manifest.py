"""Common metadata format for all source datasets."""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
from typing import Iterable


@dataclass(frozen=True)
class SampleRecord:
    """One source recording or segmented sequence."""

    sample_id: str
    dataset: str
    subject_id: str
    exercise: str
    source_path: str
    feature_path: str = ""
    num_frames: int = 0
    fps: float = 0.0
    split: str = ""
    label_type: str = "exercise"
    repetition_id: str = ""

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "SampleRecord":
        values = dict(row)
        values["num_frames"] = int(values.get("num_frames") or 0)
        values["fps"] = float(values.get("fps") or 0)
        return cls(**values)


def write_manifest(records: Iterable[SampleRecord], path: str | Path) -> None:
    """Write records as a stable CSV manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SampleRecord.__dataclass_fields__)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def read_manifest(path: str | Path) -> list[SampleRecord]:
    """Read and validate a manifest written by :func:`write_manifest`."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = set(SampleRecord.__dataclass_fields__)
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                f"Manifest columns must be exactly {sorted(expected)}; "
                f"got {reader.fieldnames}"
            )
        return [SampleRecord.from_row(row) for row in reader]
