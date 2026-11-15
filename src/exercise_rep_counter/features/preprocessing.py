"""Training-safe preprocessing for sequential feature arrays."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FeatureNormalizer:
    """Per-feature standardization fitted only on training frames."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, sequences: list[np.ndarray]) -> "FeatureNormalizer":
        if not sequences:
            raise ValueError("Cannot fit normalization without training sequences")
        feature_dim = sequences[0].shape[1]
        if any(sequence.ndim != 2 or sequence.shape[1] != feature_dim for sequence in sequences):
            raise ValueError("All sequences must be 2D with the same feature dimension")
        values = np.concatenate(sequences, axis=0).astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Training sequences contain NaN or infinite values")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        values = np.asarray(sequence, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("Sequence shape does not match fitted normalization")
        if not np.isfinite(values).all():
            raise ValueError("Sequence contains NaN or infinite values")
        return ((values - self.mean) / self.scale).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        with np.load(path) as archive:
            if set(archive.files) != {"mean", "scale"}:
                raise ValueError("Normalization archive must contain mean and scale")
            return cls(archive["mean"], archive["scale"])
