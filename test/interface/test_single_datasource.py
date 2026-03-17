import numpy as np
import pytest
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary

from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.preprocessing import TimeSplitter
from picid.interface import CustomSingleSourceLoader, EntryInterface
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


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


@pytest.mark.skip(reason="no way of currently testing training")
@pytest.mark.parametrize('get_sources', [[1, 'numpy'], [1, 'csv']], indirect=True)
def test_training(get_sources):
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "rul"],
    )

    source = get_sources[0]
    if isinstance(source, np.ndarray):
        datasource = CustomSingleSourceLoader.load_from_numpy(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='rul')
    else:
        datasource = CustomSingleSourceLoader.load_from_csv(source,
                                                        target_column=-1,
                                                        data_splitter=data_splitter,
                                                        task_mode='rul')

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                  metadata={'apply_to': 'rul', 'fit_on': 'train'})

    transforms = [scaler, scaler_target]
    interface = EntryInterface()

    processed_datasource = interface.process_datasource(datasource, transforms)

    model = LSTMConfig(n_layers=8)

    task_definition = Prognostic(task_type='rul')

    loggers = [CsvLogger(name='base_csv_logger', version='1')]

    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]

    evaluators = {s: RulEvaluatorConfig() for s in ['train', 'test', 'val']}

    interface.train(task_definition=task_definition, model=model, datasource=processed_datasource, callbacks=callbacks,
                    loggers=loggers, evaluators=evaluators, transforms=transforms)

