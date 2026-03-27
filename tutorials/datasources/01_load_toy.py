#!/usr/bin/env python3
"""Tutorial: Load data from ToyRaggedLoader and inspect SplitDatasetContainer."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.data.datasources.toy_example import ToyRaggedLoader
from picid.data.data_objects import SplitDatasetContainer


def main():
    loader = ToyRaggedLoader(
        data_dir=".", data_name="toy", task_mode="anomaly_detection"
    )
    loader.load_data()
    loader.split_data()
    container = loader.get_data()
    assert isinstance(container, SplitDatasetContainer)
    assert "train" in container.features
    assert "test" in container.features
    train_feats = container.features.train
    assert isinstance(train_feats, list)
    assert len(train_feats) > 0

    def _shape(a):
        return a.to_numpy().shape if hasattr(a, "to_numpy") else a.shape

    print(
        f"Train units: {len(train_feats)}, shapes: {[_shape(a) for a in train_feats[:3]]}"
    )
    print("OK")


if __name__ == "__main__":
    main()
