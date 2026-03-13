import dis
from io import StringIO

import dill
import hydra.utils
import lightning
import numpy as np
import pytest
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary

from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.preprocessing import TimeSplitter
from picid.interface import CustomSingleSourceLoader, EntryInterface
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


# def test_datasource_numpy():
#
#     cycles = 2  # how many sine cycles
#     resolution = 1000  # how many datapoints to generate
#     length = np.pi * 2 * cycles
#
#     my_wave = np.sin(np.arange(0, length, length / resolution))
#     dataset = np.stack((my_wave[:-1], my_wave[1:]), -1)
#
#     def get_random_datasplit(dev_ratio: float = 0.1, test_ratio: float = 0.1):
#         assert dev_ratio + test_ratio < 1, "dev_ratio + test_ratio must be less than 1."
#
#         def random_datasplit(data_dict):
#             features = data_dict['features']
#
#             idxs = np.arange(len(features))
#
#             train_split = int(len(idxs) * (1 - (dev_ratio + test_ratio)))
#             test_split = int(len(idxs) * dev_ratio)
#             dev_split = int(len(idxs) * test_ratio)
#
#             train_idx = idxs[:train_split]
#             dev_idx = idxs[train_split:train_split + dev_split]
#             test_idx = idxs[train_split + dev_split + test_split:]
#
#             return {'train': train_idx, 'val': dev_idx, 'test': test_idx}
#
#         return random_datasplit
#
#     datasource = CustomSingleSourceLoader.load_from_numpy(dataset,
#                                                           target_column=-1,
#                                                           data_name='test',
#                                                           task_mode='rul',
#                                                           data_splitter=get_random_datasplit(dev_ratio=0.2, test_ratio=0.1))
#
#     datasource.load_data()
#
#     scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
#                            metadata={'apply_to': 'features', 'fit_on': 'train'})
#     scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
#                   metadata={'apply_to': 'rul', 'fit_on': 'train'})
#     transforms = [scaler, scaler_target]
#
#     model = LSTMConfig(n_layers=4)
#
#     task_definition = Prognostic(task_type='rul')
#
#     loggers = [CsvLogger(name='base_csv_logger', version='1')]
#
#     # TODO: implement evaluators
#
#     callbacks = [EarlyStopping(monitor=None),
#                  RichModelSummary()]
#
#     interface = EntryInterface()
#     processed_datasource = interface.process_datasource(datasource, transforms)
#
#     interface.train(datasource=processed_datasource,
#                     loggers=loggers,
#                     callbacks=callbacks,
#                     task_definition=task_definition,
#                     model=model,
#                     transforms=transforms)
#
#
# # def test_multi_datasource_numpy():
# #     cycles = 2  # how many sine cycles
# #     resolution = 1000  # how many datapoints to generate
# #     length = np.pi * 2 * cycles
# #
# #     my_wave = np.sin(np.arange(0, length, length / resolution))
# #     dataset1 = np.stack((my_wave[:-1], my_wave[1:]), -1)
# #
# #     my_wave = np.sin(np.arange(0, length, length / resolution))
# #     dataset2 = np.stack((my_wave[:-1], my_wave[1:]), -1)
# #
# #     my_wave = np.sin(np.arange(0, length, length / resolution))
# #     dataset3 = np.stack((my_wave[:-1], my_wave[1:]), -1)
# #
# #     datasource = CustomMultiSourceLoader.load_from_primitive([dataset1, dataset2, dataset3],
# #                                                           target_column=-1,
# #                                                           data_name='test multi source',
# #                                                           task_mode='rul')
# #
# #     datasource.load_data()
# #
# #     scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
# #                            apply_to='features', fit_on='train')
# #     scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
# #                                   **{'apply_to': 'rul', 'fit_on': 'train'})
# #     transforms = [scaler, scaler_target]
# #
# #     model = LSTMConfig(n_layers=4)
# #
# #     task_definition = Prognostic(task_type='rul')
# #
# #     loggers = [CsvLogger(name='base_csv_logger', version='1')]
# #
# #     # TODO: implement evaluators
# #
# #     callbacks = [EarlyStopping(monitor=None),
# #                  RichModelSummary()]
# #
# #     interface = EntryInterface()
# #     processed_datasource = interface.process_datasource(datasource, transforms)
# #
# #     interface.train(datasource=processed_datasource,
# #                     loggers=loggers,
# #                     callbacks=callbacks,
# #                     task_definition=task_definition,
# #                     model=model,
# #                     transforms=transforms)
#
#
# def test_pydantic_easy_interface_input_processed_datasource():
#     interface = EntryInterface()
#
#     scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
#                            metadata={'apply_to': 'features', 'fit_on': 'train'})
#     scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
#                   metadata={'apply_to': 'rul', 'fit_on': 'train'})
#
#     transforms = [scaler, scaler_target]
#
#     datasource = interface.get_datasource('phme20')
#     processed_datasource = interface.process_datasource(datasource, transforms)
#
#     model = LSTMConfig(n_layers=4)
#
#     task_definition = Prognostic(task_type='rul')
#
#     loggers = [CsvLogger(name='base_csv_logger', version='1')]
#
#     # TODO: implement evaluators
#
#     callbacks = [EarlyStopping(monitor=None),
#                  RichModelSummary()]
#
#     interface.train(datasource=processed_datasource,
#                     loggers=loggers,
#                     callbacks=callbacks,
#                     task_definition=task_definition,
#                     model=model)


@pytest.mark.parametrize('get_sources', [[1, 'numpy'], [1, 'csv']], indirect=True)
def test_single_source_loader_standalone_load_split_get_returns_split_container(
    get_sources,
):
    """Standalone: load → split → get_data returns SplitDatasetContainer."""
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "target"],
    )

    source = get_sources[0]
    if isinstance(source, np.ndarray):
        loader = CustomSingleSourceLoader.load_from_numpy(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')
    else:
        loader = CustomSingleSourceLoader.load_from_csv(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')

    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data
    assert "target" in data
    assert "train" in data["features"]
    assert "val" in data["features"]
    assert "test" in data["features"]


@pytest.mark.parametrize('get_sources', [[1, 'numpy'], [1, 'csv']], indirect=True)
def test_single_source_loader_get_data_before_load_raises(get_sources):
    """get_data() before load_data() raises AssertionError."""
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "target"],
    )

    source = get_sources[0]
    if isinstance(source, np.ndarray):
        loader = CustomSingleSourceLoader.load_from_numpy(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')
    else:
        loader = CustomSingleSourceLoader.load_from_csv(source,
                                                        target_column=-1,
                                                        data_splitter=data_splitter,
                                                        task_mode='target')

    with pytest.raises(AssertionError, match="must be loaded"):
        loader.get_data()


@pytest.mark.parametrize('get_sources', [[1, 'numpy'], [1, 'csv']], indirect=True)
def test_single_source_loader_get_data_before_split_raises(get_sources):
    """Standalone: get_data() before split_data() raises AssertionError."""
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "target"],
    )

    source = get_sources[0]
    if isinstance(source, np.ndarray):
        loader = CustomSingleSourceLoader.load_from_numpy(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')
    else:
        loader = CustomSingleSourceLoader.load_from_csv(source,
                                                        target_column=-1,
                                                        data_splitter=data_splitter,
                                                        task_mode='target')

    loader.load_data()
    with pytest.raises(AssertionError, match="must be splitted"):
        loader.get_data()


@pytest.mark.parametrize('get_sources', [[1, 'numpy'], [1, 'csv']], indirect=True)
def test_single_source_loader_get_meta_data_after_load(get_sources):
    """get_meta_data() after load returns dict (empty for CustomSingleSourceLoader)."""
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "target"],
    )

    source = get_sources[0]
    if isinstance(source, np.ndarray):
        loader = CustomSingleSourceLoader.load_from_numpy(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')
    else:
        loader = CustomSingleSourceLoader.load_from_csv(source,
                                                        target_column=-1,
                                                        data_splitter=data_splitter,
                                                        task_mode='target')

    loader.load_data()
    loader.load_data()
    meta = loader.get_meta_data()
    assert isinstance(meta, dict)