#!/usr/bin/env python3
"""Tutorial: MinMaxScalerSklearn fit on train, transform per segment."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


def main():
    metadata = {"apply_to_keys": ["features"]}

    # Train chunks
    arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    train_chunks = [
        NamedTransformInput(features=arr1),
        NamedTransformInput(features=arr2),
    ]

    # Val and test chunks
    val_chunks = [
        NamedTransformInput(
            features=np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        ),
    ]
    test_chunks = [
        NamedTransformInput(
            features=np.array([[0.5, 1.0], [9.0, 10.0]], dtype=np.float32)
        ),
    ]

    scaler = MinMaxScalerSklearn()
    scaler.fit_multi_source(train_chunks, metadata)

    assert scaler.scaler.data_min_ is not None, "Scaler should be fitted"

    for chunk in train_chunks:
        out = scaler.transform_data(chunk, metadata)
        assert (
            out.shape == chunk["features"].shape
        ), f"Shape mismatch: {out.shape} vs {chunk['features'].shape}"

    for chunk in val_chunks:
        out = scaler.transform_data(chunk, metadata)
        assert out.shape == chunk["features"].shape

    for chunk in test_chunks:
        out = scaler.transform_data(chunk, metadata)
        assert out.shape == chunk["features"].shape

    print("OK")


if __name__ == "__main__":
    main()
