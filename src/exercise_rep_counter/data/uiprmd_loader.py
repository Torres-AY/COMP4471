"""Discovery and loading helpers for UI-PRMD skeletal files."""

from pathlib import Path
import re

import numpy as np

from .manifest import SampleRecord


SKELETON_EXTENSIONS = {".csv", ".txt", ".npy", ".npz", ".mat"}


def _find_subject(path: Path) -> str:
    match = re.search(r"(?:subject|subj|person|s)[_\-\s]?(\d+)", str(path).lower())
    return f"s{int(match.group(1)):02d}" if match else "unknown"


def _find_movement(path: Path) -> str:
    match = re.search(r"(?:movement|move|m)[_\-\s]?(\d+)", str(path).lower())
    return f"m{int(match.group(1)):02d}" if match else "unknown"


def discover_sequences(root: str | Path) -> list[SampleRecord]:
    """Discover UI-PRMD sequence files without assuming archive subfolders."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"UI-PRMD directory does not exist: {root_path}")

    records: list[SampleRecord] = []
    for sequence_path in sorted(
        path for path in root_path.rglob("*") if path.suffix.lower() in SKELETON_EXTENSIONS
    ):
        relative = sequence_path.relative_to(root_path)
        subject = _find_subject(relative)
        movement = _find_movement(relative)
        repetition_match = re.search(
            r"(?:rep|repetition|e)[_\-\s]?(\d+)", sequence_path.stem.lower()
        )
        repetition_id = str(int(repetition_match.group(1))) if repetition_match else ""
        records.append(
            SampleRecord(
                sample_id=f"uiprmd_{sequence_path.stem}",
                dataset="uiprmd",
                subject_id=subject,
                exercise=movement,
                source_path=str(relative),
                label_type="movement",
                repetition_id=repetition_id,
            )
        )
    if not records:
        raise ValueError(f"No supported skeletal files found under {root_path}")
    return records


def load_sequence(path: str | Path) -> np.ndarray:
    """Load a numeric UI-PRMD sequence from common archive formats.

    The returned array preserves the source joint layout. Mapping to the
    project's feature layout belongs in a separate preprocessing step.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        array = np.load(source)
    elif suffix == ".npz":
        archive = np.load(source)
        if len(archive.files) != 1:
            raise ValueError(f"Expected one array in {source}, found {archive.files}")
        array = archive[archive.files[0]]
    elif suffix in {".csv", ".txt"}:
        array = np.loadtxt(source, delimiter=",")
    elif suffix == ".mat":
        from scipy.io import loadmat

        arrays = {
            key: value
            for key, value in loadmat(source).items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        }
        if len(arrays) != 1:
            raise ValueError(f"Expected one numeric array in {source}, found {list(arrays)}")
        array = next(iter(arrays.values()))
    else:
        raise ValueError(f"Unsupported UI-PRMD sequence format: {source.suffix}")

    result = np.asarray(array, dtype=np.float32)
    if result.ndim < 2 or result.size == 0:
        raise ValueError(f"Sequence must be a non-empty 2D-or-higher array: {source}")
    if not np.isfinite(result).all():
        raise ValueError(f"Sequence contains NaN or infinite values: {source}")
    return result
