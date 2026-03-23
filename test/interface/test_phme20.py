import os

import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import RichModelSummary

from picid.interface import EntryInterface
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


@pytest.mark.skipif(os.environ.get('TEST_SKIP_TRAINING', True), reason="Not testing training right now.")
def test_pydantic_easy_interface():

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                  metadata={'apply_to': 'rul', 'fit_on': 'train'})

    transforms = [scaler, scaler_target]
    # transforms = []

    model = LSTMConfig(n_layers=8)

    task_definition = Prognostic(task_type='rul')

    loggers = [CsvLogger(name='base_csv_logger', version='1')]

    callbacks = [EarlyStopping(monitor='val/loss'),
                 RichModelSummary()]

    evaluators = {s: RulEvaluatorConfig() for s in ['train', 'test', 'val']}

    interface = EntryInterface()
    interface.train(run_name='test',task_definition=task_definition, model=model,
                    datasource=('phme20', 'raw'), callbacks=callbacks,
                    overrides=['trainer.max_epochs=1', f'trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}', 'trainer.devices=1'],
                    loggers=loggers, evaluators=evaluators, transforms=transforms)


@pytest.mark.skipif(os.environ.get('TEST_SKIP_TRAINING', True), reason="Not testing training right now.")
def test_pydantic_minimal_interface_train():
    interface = EntryInterface()

    interface.train(run_name='test',model='lstm',
                    task_definition=('prognostics', 'rul'),
                    datasource=('phme20', 'raw'),
                    overrides=['trainer.max_epochs=1', f'trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}', 'trainer.devices=1'],
                    # transforms=transforms,
                    debug=False)


@pytest.mark.skipif(os.environ.get('TEST_SKIP_TRAINING', True), reason="Not testing training right now.")
def test_pydantic_easy_interface_input_processed_datasource():
    interface = EntryInterface()

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

    callbacks = [EarlyStopping(monitor='val/loss'),
                 RichModelSummary()]

    interface.train(run_name='test',task_definition=task_definition, model=model, datasource=processed_datasource, callbacks=callbacks,
                    overrides=['trainer.max_epochs=1', f'trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}', 'trainer.devices=1'],
                    loggers=loggers)
