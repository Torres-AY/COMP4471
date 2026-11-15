"""Train and evaluate the initial UI-PRMD GRU classifier."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from exercise_rep_counter.models.gru import GRUClassifier
from .dataset import WindowDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def labels_from_archive(path: str | Path) -> list[str]:
    with np.load(path) as archive:
        return sorted({str(label) for label in archive["labels"].tolist()})


def evaluate(
    model: GRUClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            target = labels.to(device)
            losses.append(float(criterion(logits, target).item()))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(target.cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(targets, predictions),
        "macro_precision": precision_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def train(
    data_root: str | Path,
    checkpoint_path: str | Path,
    window_length: int = 60,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, float]:
    """Train a GRU and save the checkpoint with best validation macro-F1."""

    set_seed(seed)
    root = Path(data_root)
    train_path = root / f"train_w{window_length}.npz"
    validation_path = root / f"validation_w{window_length}.npz"
    test_path = root / f"test_w{window_length}.npz"
    label_to_id = {
        label: index for index, label in enumerate(labels_from_archive(train_path))
    }
    train_data = WindowDataset(train_path, label_to_id)
    validation_data = WindowDataset(validation_path, label_to_id)
    test_data = WindowDataset(test_path, label_to_id)
    if not len(train_data) or not len(validation_data) or not len(test_data):
        raise ValueError("Train, validation, and test datasets must all be non-empty")

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model = GRUClassifier(
        input_dim=train_data.features.shape[-1],
        num_classes=len(label_to_id),
    ).to(selected_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, float] = {}

    for _ in range(epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features.to(selected_device)), labels.to(selected_device))
            loss.backward()
            optimizer.step()
        validation = evaluate(model, validation_loader, criterion, selected_device)
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_validation = validation
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, selected_device)
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_to_id": label_to_id,
            "input_dim": train_data.features.shape[-1],
            "window_length": window_length,
            "validation_metrics": best_validation,
            "test_metrics": test_metrics,
        },
        destination,
    )
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/uiprmd"))
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        type=Path,
        default=Path("checkpoints/uiprmd_gru_best.pt"),
    )
    parser.add_argument("--window-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    metrics = train(**vars(args))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
