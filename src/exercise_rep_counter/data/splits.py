"""Subject-safe dataset splitting."""

from collections import defaultdict
import random

from .manifest import SampleRecord


def split_by_subject(
    records: list[SampleRecord],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> list[SampleRecord]:
    """Assign all records from each subject to exactly one split."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")

    subjects = sorted({record.subject_id for record in records})
    if "unknown" in subjects and len(subjects) > 1:
        raise ValueError("Cannot create a subject-independent split with unknown subjects")
    shuffled = subjects[:]
    random.Random(seed).shuffle(shuffled)
    train_end = max(1, round(len(shuffled) * train_fraction))
    validation_end = train_end + round(len(shuffled) * validation_fraction)
    assignments = {
        subject: "train" if index < train_end else
        "validation" if index < validation_end else "test"
        for index, subject in enumerate(shuffled)
    }
    return [
        SampleRecord(**{**record.__dict__, "split": assignments[record.subject_id]})
        for record in records
    ]
