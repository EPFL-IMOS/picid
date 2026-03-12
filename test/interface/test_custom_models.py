from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary
from torch import nn

from picid.interface import EasyInterface
from picid.interface.model_wrapper import CustomModelWrapper, ModelWrapper
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.model import LSTMConfig, MLPConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.model.methods.mlp import MLP
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


def test_linear_model():

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                  metadata={'apply_to': 'rul', 'fit_on': 'train'})

    transforms = [scaler, scaler_target]

    config = {
        "input_channels": 1,
        "seq_len": 16,
        "hidden_dim": 128,
        "num_layers": 3,
    }

    backbone = MLP(config, task_type='rul', num_targets=1)
    backbone = ModelWrapper(model=backbone, pre_process_function=lambda x: nn.Flatten(1)(x).permute(0, 2, 1))

    custom_model = CustomModelWrapper(task_type='rul', model=backbone)

    # model = MLPConfig(input_ch    annels=1, num_targets=1)

    task_definition = Prognostic(task_type='rul')

    loggers = [CsvLogger(name='base_csv_logger', version='1')]

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    evaluators = {s: RulEvaluatorConfig() for s in ['train', 'test', 'val']}

    interface = EasyInterface()
    interface.train(datasource='phme20',
                    # custom_backbone=custom_model,
                    loggers=loggers,
                    callbacks=callbacks,
                    task_definition=task_definition,
                    model=custom_model,
                    evaluators=evaluators,
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
