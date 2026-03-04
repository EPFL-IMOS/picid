import dis
from io import StringIO

import dill
import hydra.utils
import lightning
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary

from picid.interface import CustomSingleSourceLoader, EasyInterface
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


def test_datasource_numpy():

    cycles = 2  # how many sine cycles
    resolution = 1000  # how many datapoints to generate
    length = np.pi * 2 * cycles

    my_wave = np.sin(np.arange(0, length, length / resolution))
    dataset = np.stack((my_wave[:-1], my_wave[1:]), -1)

    def get_random_datasplit(dev_ratio: float = 0.1, test_ratio: float = 0.1):
        assert dev_ratio + test_ratio < 1, "dev_ratio + test_ratio must be less than 1."

        def random_datasplit(data_dict):
            features = data_dict['features']

            idxs = np.arange(len(features))

            train_split = int(len(idxs) * (1 - (dev_ratio + test_ratio)))
            test_split = int(len(idxs) * dev_ratio)
            dev_split = int(len(idxs) * test_ratio)

            train_idx = idxs[:train_split]
            dev_idx = idxs[train_split:train_split + dev_split]
            test_idx = idxs[train_split + dev_split + test_split:]

            return {'train': train_idx, 'val': dev_idx, 'test': test_idx}

        return random_datasplit

    datasource = CustomSingleSourceLoader.load_from_numpy(dataset,
                                                          target_column=-1,
                                                          data_name='test',
                                                          task_mode='rul',
                                                          data_splitter=get_random_datasplit(dev_ratio=0.2, test_ratio=0.1))

    datasource.load_data()

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


# def test_multi_datasource_numpy():
#     cycles = 2  # how many sine cycles
#     resolution = 1000  # how many datapoints to generate
#     length = np.pi * 2 * cycles
#
#     my_wave = np.sin(np.arange(0, length, length / resolution))
#     dataset1 = np.stack((my_wave[:-1], my_wave[1:]), -1)
#
#     my_wave = np.sin(np.arange(0, length, length / resolution))
#     dataset2 = np.stack((my_wave[:-1], my_wave[1:]), -1)
#
#     my_wave = np.sin(np.arange(0, length, length / resolution))
#     dataset3 = np.stack((my_wave[:-1], my_wave[1:]), -1)
#
#     datasource = CustomMultiSourceLoader.load_from_primitive([dataset1, dataset2, dataset3],
#                                                           target_column=-1,
#                                                           data_name='test multi source',
#                                                           task_mode='rul')
#
#     datasource.load_data()
#
#     scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
#                            apply_to='features', fit_on='train')
#     scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
#                                   **{'apply_to': 'rul', 'fit_on': 'train'})
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
#     interface = EasyInterface()
#     processed_datasource = interface.process_datasource(datasource, transforms)
#
#     interface.train(datasource=processed_datasource,
#                     loggers=loggers,
#                     callbacks=callbacks,
#                     task_definition=task_definition,
#                     model=model,
#                     transforms=transforms)


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
