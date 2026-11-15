"""GRU classifier for fixed-length feature windows."""

import torch
from torch import nn


class GRUClassifier(nn.Module):
    """Classify a temporal feature window into one movement or phase label."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError("input_dim must be positive and num_classes must exceed one")
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("Expected features with shape [batch, frames, feature_dim]")
        sequence, _ = self.gru(features)
        return self.classifier(self.dropout(sequence[:, -1]))
