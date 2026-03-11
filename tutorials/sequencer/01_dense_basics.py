#!/usr/bin/env python3
"""Tutorial: DenseArraySequencer basics."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.data.optimization.sequencer import DenseArraySequencer


def main():
    arr = np.random.randn(100, 3).astype(np.float32)
    seq = DenseArraySequencer(
        array=arr,
        seq_len=4,
        label_len=2,
        pred_len=2,
        stride=2,
    )
    batch = seq.sequences_batch([0, 1, 2])
    seq_x, seq_y = batch[0], batch[1]
    assert seq_x.shape[0] == 3
    assert seq_x.shape[1] == 4
    assert seq_y.shape[1] == 4
    print(f"seq_x: {seq_x.shape}, seq_y: {seq_y.shape}")
    print("OK")


if __name__ == "__main__":
    main()
