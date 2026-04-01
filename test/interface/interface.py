import pytest

from picid.interface import EntryInterface
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


def test_pydantic_minimal_interface_asserts():
    interface = EntryInterface()

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                                  metadata={'apply_to': 'rul', 'fit_on': 'train'})

    transforms = [scaler, scaler_target]

    with pytest.raises(AssertionError):
        interface.train(run_name='test',task_definition=('prognostics', 'lstm'),
                        model='testing_assert',
                        datasource='phme20',
                        overrides='trainer.max_epochs=1',
                        transforms=transforms)

    with pytest.raises(AssertionError):
        interface.train(run_name='test',task_definition=('testing_assert', 'lstm'),
                        model='lstm',
                        datasource='phme20',
                        overrides='trainer.max_epochs=1',
                        transforms=transforms)

    with pytest.raises(AssertionError):
        interface.train(run_name='test',task_definition=('prognostics', 'testing_assert'),
                        model='lstm',
                        datasource='phme20',
                        overrides='trainer.max_epochs=1',
                        transforms=transforms)

    with pytest.raises(AssertionError):
        interface.train(run_name='test',task_definition=('prognostics', 'rul'),
                        model='lstm',
                        evaluators='testing_assert',
                        datasource='phme20',
                        overrides='trainer.max_epochs=1',
                        transforms=transforms)

    with pytest.raises(AssertionError):
        interface.train(run_name='test', task_definition=('prognostics', 'rul'),
                        model='lstm',
                        evaluators='prova',
                        datasource=('phme20', 'raw'),
                        overrides='trainer.max_epochs=1')


