import numpy as np
import pytest
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from lmetk.data.datasets import TimeSeriesDataset
from lmetk.data.datasets.batch_dataset import BatchDataset 
import os
from pathlib import Path




@pytest.fixture(scope="module")
def dataset_val():
    
    GlobalHydra.instance().clear()
    base_dir = Path(__file__).resolve().parents[3]
    data_path = os.path.abspath(f"{base_dir}/datasets/ETT-small/ETTh1.csv")
    assert os.path.exists(data_path), f"File not found: {data_path}"

    loader_config = OmegaConf.create({
        "_target_": "lmetk.data.datasets.Dataset_ETT_hour_refactored",
        "load_arguments": {
            "data_name": "ETTh1",
            "data_folder": "ETT-small",
            "data_path": data_path,

            "flag": "train",
            "size": None,

            "features_mode": "S",
            "target_name": "OT",
            "scale": True,
            "freq": "h",

            "timestamp_name": "date",
            "time_features": ["month", "day", "weekday", "hour"],
            "normalize_time_features": False
        },
        "data_splitter": {
            "_target_": "lmetk.data.preprocessing.SimpleSplitter",
            "train": 8640,
            "val": 2880,
            "test": None,

            "seq_len": 384,
            "label_len": 96,
            "pred_len": 96
        },
        "scaler": {
            "features": {
                "_target_": "sklearn.preprocessing.StandardScaler"
            }
        }
    })

    loader = hydra.utils.instantiate(loader_config)
    loader._pipeline()  # Ensure the pipeline is executed
    splitted_data, splitted_timestamps = loader.get_data()

    ds_val = TimeSeriesDataset(
        features=splitted_data["val"],
        time_features=splitted_timestamps["val"],
        seq_len=384,
        label_len=96,
        pred_len=96,
    )
    return ds_val


def test_timeseriesdataset_first_sample(dataset_val):
    """Checks the first sample from TimeSeriesDataset matches expected output."""
    i = 0
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset_val[4]

    # Expected results (last 5 timesteps only)
    expected_seq_x = np.array([
       [0.39489377],
       [0.41788715],
       [0.37190018],    
       [0.32591301],
       [0.30291963]
       ])
    expected_seq_y = np.array([
        [0.28766311],
        [0.42562442],
        [0.43325258],
        [0.37190018],
        [0.35664366]
    ])
    expected_seq_x_mark = np.array([
        [6, 25, 6, 23],
        [6, 26, 0, 0],
        [6, 26, 0, 1],
        [6, 26, 0, 2],
        [6, 26, 0, 3]
    ])
    expected_seq_y_mark = np.array([
        [6, 29, 3, 23],
        [6, 30, 4, 0],
        [6, 30, 4, 1],
        [6, 30, 4, 2],
        [6, 30, 4, 3]
    ])

    np.testing.assert_array_almost_equal(seq_x[-5:], expected_seq_x)
    np.testing.assert_array_almost_equal(seq_y[-5:], expected_seq_y)
    np.testing.assert_array_equal(seq_x_mark[-5:], expected_seq_x_mark)
    np.testing.assert_array_equal(seq_y_mark[-5:], expected_seq_y_mark)

    print("✅ TimeSeriesDataset first sample check passed.")

def test_efficient_timeseriesdataset_first_sample(dataset_val):
    base_dir = Path(__file__).resolve().parents[3]
    data_path = os.path.abspath(f"{base_dir}/datasets/ETT-small/ETTh1.csv")
    assert os.path.exists(data_path), f"File not found: {data_path}"

    loader_config = OmegaConf.create({
        "_target_": "lmetk.data.datasets.Dataset_ETT_hour_refactored",
        "load_arguments": {
            "data_name": "ETTh1",
            "data_folder": "ETT-small",
            "data_path": data_path,

            "flag": "train",
            "size": None,

            "features_mode": "S",
            "target_name": "OT",
            "scale": True,
            "freq": "h",

            "timestamp_name": "date",
            "time_features": ["month", "day", "weekday", "hour"],
            "normalize_time_features": False
        },
        "data_splitter": {
            "_target_": "lmetk.data.preprocessing.SimpleSplitter",
            "train": 8640,
            "val": 2880,
            "test": None,

            "seq_len": 384,
            "label_len": 96,
            "pred_len": 96
        },
        "scaler": {
            "features": {
                "_target_": "sklearn.preprocessing.StandardScaler"
            }
        }
    })

    loader = hydra.utils.instantiate(loader_config)
    loader._pipeline()  # Ensure the pipeline is executed
    splitted_data, splitted_timestamps = loader.get_data()

    batch_dataset = BatchDataset(
        features=splitted_data['val'], 
        time_features=splitted_timestamps['val'], 
        seq_len=384, 
        label_len=96, 
        pred_len=96,
        stride=1,
    )
    
    return batch_dataset