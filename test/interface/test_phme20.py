import os

import pytest
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary

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

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    evaluators = {s: RulEvaluatorConfig() for s in ['train', 'test', 'val']}

    interface = EntryInterface()
    interface.train(task_definition=('prognostics', 'rul'), model=model,
                    datasource=('phme20', 'raw'), callbacks=callbacks,
                    overrides='trainer.max_epochs=1',
                    loggers=loggers, evaluators=evaluators, transforms=transforms,
                    debug=True)


def test_pydantic_minimal_interface_asserts():
    interface = EntryInterface()

    interface.train(model='lstm',
                    task_definition='prognostics',
                    datasource=('phme20', 'raw'),
                    overrides='trainer.max_epochs=1',
                    # transforms=transforms,
                    debug=True)

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                                  metadata={'apply_to': 'rul', 'fit_on': 'train'})

    transforms = [scaler, scaler_target]

    with pytest.raises(AssertionError):
        interface.train(task_definition=('prognostics', 'lstm'),
                        model='testing_assert',
                        datasource='phme20',
                        overrides='trainer.max_epochs=1',
                        transforms=transforms)

    with pytest.raises(AssertionError):
        interface.train(task_definition=('testing_assert', 'lstm'),
                        model='lstm',
                        datasource='phme20',
                        overrides='trainer.max_epochs=1',
                        transforms=transforms)

    with pytest.raises(AssertionError):
        interface.train(task_definition=('prognostics', 'testing_assert'),
                        model='lstm',
                        datasource='phme20',
                        overrides='trainer.max_epochs=1',
                        transforms=transforms)



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

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    interface.train(task_definition=task_definition, model=model, datasource=processed_datasource, callbacks=callbacks,
                    overrides='trainer.max_epochs=1',
                    loggers=loggers)
