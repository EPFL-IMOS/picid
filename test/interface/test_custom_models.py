from einops import rearrange
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary
from torch import nn

from picid.interface import EntryInterface
from picid.interface.model import CustomModelTrainer
from picid.interface.model.wrapper import ModelWrapper
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.model import LSTMConfig, MLPConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.model.methods.cnn_1d import EncoderModel
from picid.model.methods.mlp import MLP
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


def test_mlp_model():

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                  metadata={'apply_to': 'rul', 'fit_on': 'train'})

    transforms = [scaler, scaler_target]

    task_definition = Prognostic(task_type='rul', pred_len=0)

    config = {
        "input_channels": 5,
        "seq_len": task_definition.seq_len,
        "hidden_dim": 64,
        "num_layers": 3,
    }

    backbone = MLP(config, task_type=task_definition.task_type, num_targets=1)

    backbone = ModelWrapper(model=backbone,
                            pre_process_function=lambda x: x.permute(0, 2, 1))

    custom_model = CustomModelTrainer(task_type=task_definition.task_type,
                                      model=backbone)

    loggers = [CsvLogger(name='base_csv_logger', version='1')]

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    evaluators = {s: RulEvaluatorConfig() for s in ['train', 'test', 'val']}

    interface = EntryInterface()
    datasource = interface.get_datasource('phme20')
    processed_datasource = interface.process_datasource(datasource, transforms)

    interface.train(datasource=processed_datasource,
                    # custom_backbone=custom_model,
                    loggers=loggers,
                    callbacks=callbacks,
                    task_definition=task_definition,
                    model=custom_model,
                    evaluators=evaluators,
                    transforms=transforms)


def test_1dcnn_model():
    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                                  metadata={'apply_to': 'rul', 'fit_on': 'train'})

    transforms = [scaler, scaler_target]

    task_definition = Prognostic(task_type='rul')

    config = {
        "input_channels": 5,
        "latent_dim": 128,
        "dropout_prob": 0.2,
        "output_channels": [16, 32, 64, 64, 64],
        "kernels": [5, 5, 5, 5, 5],
        "strides": [1, 1, 1, 1, 1],
        "dilations": [1, 2, 4, 8, 1],
        "input_seq_len": task_definition.seq_len,
    }

    backbone = EncoderModel(
        config=config,
        task_type=task_definition.task_type,
        num_classes=1,
    )

    backbone = ModelWrapper(model=backbone,
                            pre_process_function=lambda x: x.permute(0, 2, 1))

    custom_model = CustomModelTrainer(task_type=task_definition.task_type,
                                      model=backbone)

    loggers = [CsvLogger(name='base_csv_logger', version='1')]

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    evaluators = {s: RulEvaluatorConfig() for s in ['train', 'test', 'val']}

    interface = EntryInterface()
    datasource = interface.get_datasource('phme20')
    processed_datasource = interface.process_datasource(datasource, transforms)

    interface.train(datasource=processed_datasource,
                    # custom_backbone=custom_model,
                    loggers=loggers,
                    callbacks=callbacks,
                    task_definition=task_definition,
                    model=custom_model,
                    evaluators=evaluators,
                    transforms=transforms)
