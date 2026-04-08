import os
from copy import deepcopy

import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.preprocessing import TimeSplitter, BySourceSplitter
from picid.interface.datasources import CustomMultiSourceLoader


@pytest.mark.parametrize("get_sources", [[3, "numpy"], [3, "csv"]], indirect=True)
@pytest.mark.unit
def test_multi_source_loader_not_multi_part_load_split_get_returns_split_container(
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

    sources = {str(i): d for i, d in enumerate(get_sources)}
    splitters = {str(i): deepcopy(data_splitter) for i, d in enumerate(get_sources)}

    datasource = CustomMultiSourceLoader.load_from_primitive(
        sources,
        target_column=-1,
        data_name="test multi source",
        task_mode="target",
        data_splitter=splitters,
    )

    datasource.load_data()
    datasource.split_data()
    data = datasource.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data
    assert "target" in data
    assert "train" in data["features"]
    assert "val" in data["features"]
    assert "test" in data["features"]


@pytest.mark.parametrize("get_sources", [[3, "numpy"], [3, "csv"]], indirect=True)
@pytest.mark.unit
def test_multi_source_loader_multi_part_load_split_get_returns_split_container(
    get_sources,
):
    """Standalone: load → split → get_data returns SplitDatasetContainer."""

    sources = {str(i): d for i, d in enumerate(get_sources)}

    splitter = BySourceSplitter(
        sources_train=[str(0)], sources_val=[str(1)], sources_test=[str(2)]
    )

    datasource = CustomMultiSourceLoader.load_from_primitive(
        sources,
        target_column=-1,
        data_name="test multi source",
        task_mode="target",
        data_splitter=splitter,
    )

    datasource.load_data()
    datasource.split_data()
    data = datasource.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data
    assert "target" in data
    assert "train" in data["features"]
    assert "val" in data["features"]
    assert "test" in data["features"]


@pytest.mark.skipif(
    os.environ.get("TEST_SKIP_TRAINING", True), reason="Not testing training right now."
)
@pytest.mark.integration
@pytest.mark.optional_dep
@pytest.mark.parametrize("get_sources", [[3, "numpy"], [3, "csv"]], indirect=True)
def test_training(get_sources):
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

    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "rul"],
    )

    sources = {str(i): d for i, d in enumerate(get_sources)}
    splitters = {str(i): deepcopy(data_splitter) for i, d in enumerate(get_sources)}

    datasource = CustomMultiSourceLoader.load_from_primitive(
        sources,
        target_column=-1,
        data_name="test multi source",
        task_mode="rul",
        data_splitter=splitters,
    )

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
    interface = EntryInterface()

    processed_datasource = interface.process_datasource(datasource, transforms)

    model = LSTMConfig(n_layers=8)

    task_definition = Prognostic(task_type="rul")

    loggers = [CsvLogger(name="base_csv_logger", version="1")]

    callbacks = [EarlyStopping(monitor="val/loss"), RichModelSummary()]

    evaluators = {s: RulEvaluatorConfig() for s in ["train", "test", "val"]}

    interface.train(
        run_name="test",
        task_definition=task_definition,
        model=model,
        datasource=processed_datasource,
        callbacks=callbacks,
        overrides=[
            "trainer.max_epochs=1",
            f"trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}",
            "trainer.devices=1",
        ],
        loggers=loggers,
        evaluators=evaluators,
        transforms=transforms,
    )
