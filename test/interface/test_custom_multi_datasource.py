import dis
from io import StringIO

import dill
import hydra.utils
import lightning
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary

from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.preprocessing import TimeSplitter, BySourceSplitter
from picid.interface import CustomSingleSourceLoader, EasyInterface
from picid.interface.datasources import CustomMultiSourceLoader
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


def test_multi_datasource_loading_numpy():
    train, val, test = 0.5, 0.25, 0.25

    splitter_cfg = {
        "train": train,
        "val": val,
        "test": None,  # ratio-based: test = n - train - val
        "seq_len": 10,
        "pred_len": 5,
        "create_splits_for": ["features", "timestamps", "rul"],
    }

    splitters = [TimeSplitter(**splitter_cfg), TimeSplitter(**splitter_cfg), TimeSplitter(**splitter_cfg)]

    cycles = 2  # how many sine cycles
    resolution = 1000  # how many datapoints to generate
    length = np.pi * 2 * cycles

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset1 = np.stack((my_wave[:-1], my_wave[1:]), -1)

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset2 = np.stack((my_wave[:-1], my_wave[1:]), -1)

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset3 = np.stack((my_wave[:-1], my_wave[1:]), -1)

    datasource = CustomMultiSourceLoader.load_from_primitive([dataset1, dataset2, dataset3],
                                                          target_column=-1,
                                                          data_name='test multi source',
                                                          task_mode='rul',
                                                          single_source_data_splitter=splitters)

    datasource.load_data()
    datasource.split_data()
    data = datasource.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data and "rul" in data
    for split in ["train", "val", "test"]:
        assert split in data["features"]
        # Combined from both sources: list of arrays
        assert len(data["features"][split]) >= 1

    return
    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                  metadata={'apply_to': 'rul', 'fit_on': 'train'})
    transforms = [scaler, scaler_target]

    model = LSTMConfig(n_layers=4)

    task_definition = Prognostic(task_type='rul')

    loggers = [CsvLogger(name='base_csv_logger', version='1')]

    # TODO: implement evaluators

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    interface = EasyInterface()
    processed_datasource = interface.process_datasource(datasource, transforms)

    interface.train(datasource=processed_datasource,
                    loggers=loggers,
                    callbacks=callbacks,
                    task_definition=task_definition,
                    model=model,
                    transforms=transforms)


def test_multi_datasource_processing_numpy():
    train, val, test = 0.5, 0.25, 0.25

    splitter_cfg = {
        "train": train,
        "val": val,
        "test": None,  # ratio-based: test = n - train - val
        "seq_len": 10,
        "pred_len": 5,
        "create_splits_for": ["features", "timestamps", "rul"],
    }

    cycles = 2  # how many sine cycles
    resolution = 1000  # how many datapoints to generate
    length = np.pi * 2 * cycles

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset1 = np.stack((my_wave[:-1], my_wave[1:]), -1)

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset2 = np.stack((my_wave[:-1], my_wave[1:]), -1)

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset3 = np.stack((my_wave[:-1], my_wave[1:]), -1)

    # add names or use a dictionary with names
    # use such names for BySourceSplitter

    sources = {str(i): d for i, d in enumerate([dataset1, dataset2, dataset3])}
    splitters = {str(i): TimeSplitter(**splitter_cfg) for i, d in enumerate([dataset1, dataset2, dataset3])}

    datasource = CustomMultiSourceLoader.load_from_primitive(sources,
                                                             target_column=-1,
                                                             data_name='test multi source',
                                                             task_mode='rul',
                                                             data_splitter=splitters)

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                                  metadata={'apply_to': 'rul', 'fit_on': 'train'})
    transforms = [scaler, scaler_target]

    model = LSTMConfig(n_layers=4)

    task_definition = Prognostic(task_type='rul')

    loggers = [CsvLogger(name='base_csv_logger', version='1')]

    # TODO: implement evaluators

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    interface = EasyInterface()
    processed_datasource = interface.process_datasource(datasource, transforms)

    interface.train(datasource=processed_datasource,
                    loggers=loggers,
                    callbacks=callbacks,
                    task_definition=task_definition,
                    model=model,
                    transforms=transforms)


def test_multi_datasource_single_processing_numpy():
    train, val, test = 0.5, 0.25, 0.25

    cycles = 2  # how many sine cycles
    resolution = 1000  # how many datapoints to generate
    length = np.pi * 2 * cycles

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset1 = np.stack((my_wave[:-1], my_wave[1:]), -1)

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset2 = np.stack((my_wave[:-1], my_wave[1:]), -1)

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset3 = np.stack((my_wave[:-1], my_wave[1:]), -1)

    # add names or use a dictionary with names
    # use such names for BySourceSplitter

    sources = {str(i): d for i, d in enumerate([dataset1, dataset2, dataset3])}

    splitter =  BySourceSplitter(
        sources_train = [str(0)],
            sources_val= [str(1)],
            sources_test= [str(2)])

    datasource = CustomMultiSourceLoader.load_from_primitive(sources,
                                                             target_column=-1,
                                                             data_name='test multi source',
                                                             task_mode='rul',
                                                             data_splitter=splitter)

    datasource.load_data()
    datasource.split_data()
    data = datasource.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data and "rul" in data
    for split in ["train", "val", "test"]:
        assert split in data["features"]
        # Combined from both sources: list of arrays
        assert len(data["features"][split]) >= 1

    return

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                                  metadata={'apply_to': 'rul', 'fit_on': 'train'})
    transforms = [scaler, scaler_target]

    model = LSTMConfig(n_layers=4)

    task_definition = Prognostic(task_type='rul')

    loggers = [CsvLogger(name='base_csv_logger', version='1')]

    # TODO: implement evaluators

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    interface = EasyInterface()
    processed_datasource = interface.process_datasource(datasource, transforms)

    interface.train(datasource=processed_datasource,
                    loggers=loggers,
                    callbacks=callbacks,
                    task_definition=task_definition,
                    model=model,
                    transforms=transforms)


def test_pydantic_easy_interface_input_processed_datasource():
    interface = EasyInterface()

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                  metadata={'apply_to': 'rul', 'fit_on': 'train'})

    transforms = [scaler, scaler_target]

    datasource = interface.get_datasource('phme20')
    processed_datasource = interface.process_datasource(datasource, transforms)

    model = LSTMConfig(n_layers=4)

    task_definition = Prognostic(task_type='rul')

    loggers = [CsvLogger(name='base_csv_logger', version='1')]

    # TODO: implement evaluators

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    interface.train(datasource=processed_datasource,
                    loggers=loggers,
                    callbacks=callbacks,
                    task_definition=task_definition,
                    model=model)
