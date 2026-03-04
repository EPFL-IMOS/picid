from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary

from picid.interface import EasyInterface
from picid.interface.schemas import LSTMConfig, Prognostic
from picid.interface.schemas.loggers import CsvLogger
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


def test_pydantic_easy_interface():

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
    interface.train(datasource='phme20',
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
