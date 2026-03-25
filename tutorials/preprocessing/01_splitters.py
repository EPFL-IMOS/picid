#!/usr/bin/env python3
"""Tutorial: TimeSplitter and BySourceSplitter basics."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.data.split_strategies import BySourceSplitter, TimeSplitter


def main():
    """Run the splitter tutorial smoke example."""
    # --- TimeSplitter ---
    data = np.random.randn(500, 3).astype(np.float32)
    splitter = TimeSplitter(
        train=0.6,
        val=0.2,
        test=None,  # test=None: remainder becomes test
        seq_len=4,
        label_len=2,
        pred_len=2,
    )
    splits_dict, masks = splitter.get_splits(data)
    assert "train" in splits_dict and "val" in splits_dict and "test" in splits_dict
    assert all(isinstance(v, tuple) and len(v) == 2 for v in splits_dict.values())

    splitted_data, split_masks = splitter.split_data({"features": data}, "features")
    assert (
        "train" in splitted_data and "val" in splitted_data and "test" in splitted_data
    )
    assert all(isinstance(splitted_data[k], np.ndarray) for k in splitted_data)

    # --- BySourceSplitter ---
    arr_a = np.random.randn(10, 3).astype(np.float32)
    arr_b = np.random.randn(10, 3).astype(np.float32)
    arr_c = np.random.randn(10, 3).astype(np.float32)
    data_list = [
        {"features": arr_a},
        {"features": arr_b},
        {"features": arr_c},
    ]
    source_names = ["src1", "src2", "src3"]
    splitter2 = BySourceSplitter(
        sources_train=["src1"],
        sources_val=["src2"],
        sources_test=["src3"],
    )
    result = splitter2.split_data(data_list, source_names)
    assert "features" in result
    assert (
        "train" in result["features"]
        and "val" in result["features"]
        and "test" in result["features"]
    )
    assert result["features"]["train"] is not None
    assert result["features"]["val"] is not None
    assert result["features"]["test"] is not None

    print("OK")


if __name__ == "__main__":
    main()
