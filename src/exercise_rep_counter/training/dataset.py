"""PyTorch datasets for saved temporal windows."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load one split of a compressed temporal-window archive."""

    def __init__(self, path: str | Path, label_to_id: dict[str, int]) -> None:
        with np.load(path) as archive:
            required = {"features", "labels"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            features = np.asarray(archive["features"], dtype=np.float32)
            labels = [str(label) for label in archive["labels"].tolist()]
        if features.ndim != 3:
            raise ValueError("Features must have shape [samples, frames, feature_dim]")
        unknown = sorted(set(labels) - set(label_to_id))
        if unknown:
            raise ValueError(f"Unknown labels in {path}: {unknown}")
        self.features = torch.from_numpy(features)
        self.labels = torch.tensor([label_to_id[label] for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
