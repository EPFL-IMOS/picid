#!/usr/bin/env python3
"""Tutorial: Build and inspect SplitDatasetContainer."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.data.data_objects import SplitDatasetContainer


def main():
    arr1 = np.random.randn(100, 3).astype(np.float32)
    arr2 = np.random.randn(80, 3).astype(np.float32)
    t1 = np.zeros((100, 1), dtype=np.float32)
    t2 = np.ones((80, 1), dtype=np.float32)
    container = SplitDatasetContainer(
        features={"train": [arr1, arr2], "test": [arr1[:50]]},
        target={"train": [t1, t2], "test": [t1[:50]]},
    )
    assert container.features.train[0].shape == (100, 3)
    d = container.to_split_dict()
    assert "train" in d and "features" in d["train"]
    g = container.group_by_split()
    assert len(g["train"]) == 2
    print("OK")


if __name__ == "__main__":
    main()
