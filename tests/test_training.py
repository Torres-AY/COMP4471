import numpy as np
import torch

from exercise_rep_counter.models.gru import GRUClassifier
from exercise_rep_counter.training.dataset import WindowDataset


def test_gru_output_shape():
    model = GRUClassifier(input_dim=6, num_classes=3)

    logits = model(torch.zeros((4, 10, 6)))

    assert logits.shape == (4, 3)


def test_window_dataset_uses_shared_label_mapping(tmp_path):
    path = tmp_path / "windows.npz"
    np.savez(
        path,
        features=np.zeros((2, 4, 3), dtype=np.float32),
        labels=np.array(["m02", "m01"]),
    )

    dataset = WindowDataset(path, {"m01": 0, "m02": 1})

    assert dataset.labels.tolist() == [1, 0]
