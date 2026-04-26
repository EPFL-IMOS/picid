import os

import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import RichModelSummary


from picid.interface import EntryInterface
from picid.interface.model import CustomModelTrainer
from picid.interface.model.wrapper import ModelWrapper
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.task_definition import Prognostic
from picid.model.estimators.cnn1d.model import EncoderModel
from picid.model.estimators.mlp.model import MLP
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


@pytest.mark.skipif(
    os.environ.get("TEST_SKIP_TRAINING", True), reason="Not testing training right now."
)
def test_mlp_model():
    scaler = DataTransform(
        transform_name="scaler_features",
        transform=MinMaxScalerSklearn(),
        metadata={"apply_to": "features", "fit_on": "train"},
    )
    scaler_target = DataTransform(
        transform_name="scaler_targets",
        transform=MinMaxScalerSklearn(),
        metadata={"apply_to": "rul", "fit_on": "train"},
    )

    transforms = [scaler, scaler_target]

    task_definition = Prognostic(task_type="rul", pred_len=0)

    config = {
        "input_channels": 5,
        "seq_len": task_definition.seq_len,
        "hidden_dim": 64,
        "num_layers": 3,
    }

    backbone = MLP(config, task_type=task_definition.task_type, num_targets=1)

    backbone = ModelWrapper(
        model=backbone, pre_process_function=lambda x: x.permute(0, 2, 1)
    )

    custom_model = CustomModelTrainer(
        task_type=task_definition.task_type, model=backbone
    )

    loggers = [CsvLogger(name="base_csv_logger", version="1")]

    callbacks = [EarlyStopping(monitor="val/loss"), RichModelSummary()]

    evaluators = {s: RulEvaluatorConfig() for s in ["train", "test", "val"]}

    interface = EntryInterface()
    datasource = interface.get_datasource("phme20")
    processed_datasource = interface.process_datasource(datasource, transforms)

    interface.train(
        run_name="test",
        task_definition=task_definition,
        model=custom_model,
        datasource=processed_datasource,
        callbacks=callbacks,
        loggers=loggers,
        evaluators=evaluators,
        transforms=transforms,
        overrides=[
            "trainer.max_epochs=1",
            f'trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}',
            "trainer.devices=1",
        ],
    )


@pytest.mark.skipif(
    os.environ.get("TEST_SKIP_TRAINING", True), reason="Not testing training right now."
)
def test_1dcnn_model():
    scaler = DataTransform(
        transform_name="scaler_features",
        transform=MinMaxScalerSklearn(),
        metadata={"apply_to": "features", "fit_on": "train"},
    )
    scaler_target = DataTransform(
        transform_name="scaler_targets",
        transform=MinMaxScalerSklearn(),
        metadata={"apply_to": "rul", "fit_on": "train"},
    )

    transforms = [scaler, scaler_target]

    task_definition = Prognostic(task_type="rul")

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

    backbone = ModelWrapper(
        model=backbone, pre_process_function=lambda x: x.permute(0, 2, 1)
    )

    custom_model = CustomModelTrainer(
        task_type=task_definition.task_type, model=backbone
    )

    loggers = [CsvLogger(name="base_csv_logger", version="1")]

    callbacks = [EarlyStopping(monitor="val/loss"), RichModelSummary()]

    evaluators = {s: RulEvaluatorConfig() for s in ["train", "test", "val"]}

    interface = EntryInterface()
    datasource = interface.get_datasource("phme20")
    processed_datasource = interface.process_datasource(datasource, transforms)

    interface.train(
        run_name="test",
        task_definition=task_definition,
        model=custom_model,
        datasource=processed_datasource,
        callbacks=callbacks,
        loggers=loggers,
        evaluators=evaluators,
        transforms=transforms,
        overrides=[
            "trainer.max_epochs=1",
            f'trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}',
            "trainer.devices=1",
        ],
    )
