"""Shared defaults for the initial training experiments."""

from dataclasses import dataclass
from pathlib import Path


PHASE_LABELS = ("up", "down", "transition")
EXERCISE_LABELS = ("pushup", "situp")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the first subject-independent model runs."""

    window_lengths: tuple[int, ...] = (30, 45, 60, 90)
    stride: int = 5
    target_fps: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    seed: int = 42


def repository_root() -> Path:
    """Return the repository root when installed from the source tree."""

    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the local data root used by extraction and training scripts."""

    return repository_root() / "data"
