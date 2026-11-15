"""Deterministic splits for datasets without subject identifiers."""

import random

from .manifest import SampleRecord


def split_by_record(
    records: list[SampleRecord],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> list[SampleRecord]:
    """Assign each recording to exactly one split."""

    if not records:
        raise ValueError("Cannot split an empty record list")
    if train_fraction <= 0 or validation_fraction < 0:
        raise ValueError("Split fractions must be non-negative and include training data")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    train_end = max(1, round(len(shuffled) * train_fraction))
    validation_end = train_end + round(len(shuffled) * validation_fraction)
    assignments = (
        ["train"] * train_end
        + ["validation"] * (validation_end - train_end)
        + ["test"] * (len(shuffled) - validation_end)
    )
    return [
        SampleRecord(**{**record.__dict__, "split": split})
        for record, split in zip(shuffled, assignments)
    ]
