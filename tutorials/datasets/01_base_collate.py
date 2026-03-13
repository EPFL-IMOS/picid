#!/usr/bin/env python3
"""Tutorial: BaseDataset and collate_fn for simple tabular data."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.data.datasets.base import BaseDataset


class SimpleTabularDataset(BaseDataset):
    """Minimal dataset extending BaseDataset for features (N, D) and target (N,)."""

    def __init__(self, data_dict):
        super().__init__(data_dict)
        self.features = np.asarray(data_dict["features"])
        self.target = np.asarray(data_dict["target"])
        assert len(self.features) == len(
            self.target
        ), "features and target must have same length"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.as_tensor(self.features[idx], dtype=torch.float32),
            "target": torch.as_tensor(self.target[idx], dtype=torch.float32),
        }

    def get_collate_fn(self):
        def collate(batch):
            return {
                "features": torch.stack([b["features"] for b in batch]),
                "target": torch.stack([b["target"] for b in batch]),
            }

        return collate


def main():
    N, D = 12, 4
    data_dict = {
        "features": np.random.randn(N, D).astype(np.float32),
        "target": np.random.randn(N).astype(np.float32),
    }
    dataset = SimpleTabularDataset(data_dict)
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=dataset.get_collate_fn(),
    )
    batch = next(iter(loader))
    assert batch["features"].shape[0] == 4
    assert batch["target"].shape[0] == 4
    print("OK")


if __name__ == "__main__":
    main()
