import os
import numpy as np
import pytest
import hydra
from omegaconf import OmegaConf

from lmetk.data.datasets import SWDataset, SWBatchDataset
from lmetk.data.preprocessing.single_source_preprocessor import PreProcessor
from pathlib import Path
from lmetk.transforms.transform_manager import ConfigTransformManager


@pytest.fixture(scope="module")
def dataset_val(path_cfg: str):
    #GlobalHydra.instance().clear()
    #base_dir = Path(__file__).resolve().parents[3]
    #paths_cfg = OmegaConf.load(f"{base_dir}/configs/paths/{path_cfg}.yaml")

    cfg = hydra.compose(
        config_name="test.yaml",
        overrides=[
            "datasource=ETTh1_timepfn",
            f"paths={path_cfg}",
        ],
        return_hydra_config=True,
    )
    paths_cfg = cfg.paths

    data_path = Path(f"{paths_cfg.data_dir}/ETT-small/ETTh1.csv").expanduser()
    assert os.path.exists(data_path), f"File not found: {data_path}"
    
    loader_config = OmegaConf.create(
        {
            "_target_": "lmetk.data.datasources.ETThLoader",
            
                "data_name": "ETTh1",
                "data_folder": "ETT-small",
                "data_path": data_path,
                "task_mode": "univariate",
                "target_name": "OT",
                "timestamp_name": "date",
                
                
            "data_splitter": {
                "_target_": "lmetk.data.preprocessing.TimeSplitter",
                "splitter_type": 'TimeSplitter',
                "train": 8640,
                "val": 2880,
                "test": 2880,
                "seq_len": 384,
                "label_len": 96,
                "pred_len": 96,
            },
        }
    )

    transforms_configs =  OmegaConf.create({
                "scaler": {
                    "transform": {
                        "_target_": "lmetk.transforms.base_transform.SklearnStandardScalerAdapter"
                    },
                    "metadata": {
                        "apply_to": "features",
                        "fit_on": "train"
                    }
                },
                "time_features_transform": {
                    "transform": {
                        "_target_": "lmetk.transforms.timefeatures.TimePFNTimeFeaturesTransform",
                        "freq": "h"
                    },
                    "metadata": {
                        "apply_to": "timestamps"
                    }
                }
            })

    
    dataset_source = hydra.utils.instantiate(loader_config)
    
    # Create manager
    transforms_manager = ConfigTransformManager(transforms_config=transforms_configs)

    preprocessor = PreProcessor(datasource=dataset_source, transforms=transforms_manager)
    preprocessor.pipeline()

    data_dict = preprocessor.get_processed_data()

    features = data_dict['features']
    time_features = data_dict['timestamps']
    

    ds_val = SWDataset(
        features=features["val"],
        time_features=time_features["val"],
        seq_len=384,
        label_len=96,
        pred_len=96,
    )
    return ds_val


def test_SWDataset_first_sample(dataset_val):
    """Checks the first sample from SWDataset matches expected output."""
    i = 0
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset_val[4]

    # Expected results (last 5 timesteps only)
    expected_seq_x = np.array(
        [[0.39489377], [0.41788715], [0.37190018], [0.32591301], [0.30291963]]
    )
    expected_seq_y = np.array(
        [[0.28766311], [0.42562442], [0.43325258], [0.37190018], [0.35664366]]
    )
    expected_seq_x_mark = np.array(
        [[6, 25, 6, 23], [6, 26, 0, 0], [6, 26, 0, 1], [6, 26, 0, 2], [6, 26, 0, 3]]
    )
    expected_seq_y_mark = np.array(
        [[6, 29, 3, 23], [6, 30, 4, 0], [6, 30, 4, 1], [6, 30, 4, 2], [6, 30, 4, 3]]
    )

    np.testing.assert_array_almost_equal(seq_x[-5:], expected_seq_x, err_msg= "expected_seq_x is wrong")
    np.testing.assert_array_almost_equal(seq_y[-5:], expected_seq_y, err_msg="expected_seq_y is wrong")
    # np.testing.assert_array_equal(seq_x_mark[-5:], expected_seq_x_mark, err_msg= "expected_seq_x_mark is wrong")
    # np.testing.assert_array_equal(seq_y_mark[-5:], expected_seq_y_mark, err_msg="expected_seq_y_mark is wrong")

    print("✅ SWDataset first sample check passed.")


def test_efficient_SWDataset_first_sample(dataset_val, path_cfg: str):
    # base_dir = Path(__file__).resolve().parents[3]
    # data_path = os.path.abspath(f"{base_dir}/datasets/ETT-small/ETTh1.csv")
    # assert os.path.exists(data_path), f"File not found: {data_path}"


    cfg = hydra.compose(
        config_name="test.yaml",
        overrides=[
            "datasource=ETTh1_timepfn",
            f"paths={path_cfg}",
        ],
        return_hydra_config=True,
    )
    paths_cfg = cfg.paths

    data_path = Path(f"{paths_cfg.data_dir}/ETT-small/ETTh1.csv").expanduser()
    assert os.path.exists(data_path), f"File not found: {data_path}"

    loader_config = OmegaConf.create(
        {
            "_target_": "lmetk.data.datasources.ETThLoader",
            
                "data_name": "ETTh1",
                "data_folder": "ETT-small",
                "data_path": data_path,
                "task_mode": "univariate",
                "target_name": "OT",
                "timestamp_name": "date",
                
                
            "data_splitter": {
                "_target_": "lmetk.data.preprocessing.TimeSplitter",
                "splitter_type": 'TimeSplitter',
                "train": 8640,
                "val": 2880,
                "test": 2880,
                "seq_len": 384,
                "label_len": 96,
                "pred_len": 96,
            },
        }
    )
    transforms_configs =  OmegaConf.create({
                "scaler": {
                    "transform": {
                        "_target_": "lmetk.transforms.base_transform.SklearnStandardScalerAdapter"
                    },
                    "metadata": {
                        "apply_to": "features",
                        "fit_on": "train"
                    }
                },
                "time_features_transform": {
                    "transform": {
                        "_target_": "lmetk.transforms.timefeatures.TimePFNTimeFeaturesTransform",
                        "freq": "h"
                    },
                    "metadata": {
                        "apply_to": "timestamps"
                    }
                }
            })

    dataset_source = hydra.utils.instantiate(loader_config)
    
    # Create manager
    transforms_manager = ConfigTransformManager(transforms_config=transforms_configs)

    preprocessor = PreProcessor(datasource=dataset_source, transforms=transforms_manager)
    preprocessor.pipeline()

    data_dict = preprocessor.get_processed_data()

    features = data_dict['features']
    time_features = data_dict['timestamps']

    batch_dataset = SWBatchDataset(
        features=features["val"],
        time_features=time_features["val"],
        seq_len=384,
        label_len=96,
        pred_len=96,
        stride=1,
    )

    return batch_dataset
