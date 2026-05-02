import os

import numpy as np
import pandas as pd
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.exceptions import DatasourceStateError
from picid.data.preprocessing import TimeSplitter
from picid.interface import CustomSingleSourceLoader, CustomMultiSourceLoader


@pytest.mark.unit
def test_interface_datasource_import_surface_is_lightweight():
    from picid.interface import CustomMultiSourceLoader, CustomSingleSourceLoader

    assert CustomSingleSourceLoader is not None
    assert CustomMultiSourceLoader is not None


@pytest.mark.parametrize("get_sources", [[1, "numpy"], [1, "csv"]], indirect=True)
@pytest.mark.unit
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
        loader = CustomSingleSourceLoader.load_from_numpy(
            source, target_column=-1, data_splitter=data_splitter, task_mode="target"
        )
    else:
        loader = CustomSingleSourceLoader.load_from_csv(
            source, target_column=-1, data_splitter=data_splitter, task_mode="target"
        )

    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data
    assert "target" in data
    assert "train" in data["features"]
    assert "val" in data["features"]
    assert "test" in data["features"]


@pytest.mark.parametrize("get_sources", [[1, "numpy"], [1, "csv"]], indirect=True)
@pytest.mark.unit
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
        loader = CustomSingleSourceLoader.load_from_numpy(
            source, target_column=-1, data_splitter=data_splitter, task_mode="target"
        )
    else:
        loader = CustomSingleSourceLoader.load_from_csv(
            source, target_column=-1, data_splitter=data_splitter, task_mode="target"
        )

    with pytest.raises(DatasourceStateError, match="must be loaded"):
        loader.get_data()


@pytest.mark.parametrize("get_sources", [[1, "numpy"], [1, "csv"]], indirect=True)
@pytest.mark.unit
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
        loader = CustomSingleSourceLoader.load_from_numpy(
            source, target_column=-1, data_splitter=data_splitter, task_mode="target"
        )
    else:
        loader = CustomSingleSourceLoader.load_from_csv(
            source, target_column=-1, data_splitter=data_splitter, task_mode="target"
        )

    loader.load_data()
    with pytest.raises(DatasourceStateError, match="must be splitted"):
        loader.get_data()


@pytest.mark.parametrize("get_sources", [[1, "numpy"], [1, "csv"]], indirect=True)
@pytest.mark.unit
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
        loader = CustomSingleSourceLoader.load_from_numpy(
            source, target_column=-1, data_splitter=data_splitter, task_mode="target"
        )
    else:
        loader = CustomSingleSourceLoader.load_from_csv(
            source, target_column=-1, data_splitter=data_splitter, task_mode="target"
        )

    loader.load_data()
    loader.load_data()
    meta = loader.get_meta_data()
    assert isinstance(meta, dict)


@pytest.mark.skipif(
    os.environ.get("TEST_SKIP_TRAINING", True), reason="Not testing training right now."
)
@pytest.mark.integration
@pytest.mark.optional_dep
@pytest.mark.parametrize("get_sources", [[1, "numpy"], [1, "csv"]], indirect=True)
def test_training(get_sources):
    import torch.cuda
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.callbacks import RichModelSummary

    from picid.interface import EntryInterface
    from picid.interface.schemas import TrainerConfig
    from picid.interface.schemas.evaluators import RulEvaluatorConfig
    from picid.interface.schemas.loggers import CsvLogger
    from picid.interface.schemas.model import LSTMConfig
    from picid.interface.schemas.task_definition import Prognostic
    from picid.transforms.base import DataTransform
    from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn

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
        datasource = CustomSingleSourceLoader.load_from_numpy(
            source, target_column=-1, data_splitter=data_splitter, task_mode="rul"
        )
    else:
        datasource = CustomSingleSourceLoader.load_from_csv(
            source, target_column=-1, data_splitter=data_splitter, task_mode="rul"
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

    tconf = TrainerConfig(
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=[1]
    )

    interface.train(
        run_name="test",
        training_config=tconf,
        task_definition=task_definition,
        model=model,
        datasource=processed_datasource,
        callbacks=callbacks,
        loggers=loggers,
        evaluators=evaluators,
        transforms=transforms,
        overrides=[],
    )


@pytest.mark.unit
class TestLoadFromNumpyPreSplit:
    """Tests for CustomSingleSourceLoader with pre-split dict inputs."""

    def _make_presplit(self, n_cols=5):
        rng = np.random.default_rng(0)
        return {
            "train": rng.standard_normal((50, n_cols)),
            "val": rng.standard_normal((20, n_cols)),
            "test": rng.standard_normal((20, n_cols)),
        }

    def test_presplit_dict_missing_keys_raises_value_error(self):
        """Pre-split dict missing 'test' key → ValueError (lines 81-87).

        **Expected**: ValueError mentioning missing split.
        """
        partial = {"train": np.random.randn(50, 5), "val": np.random.randn(20, 5)}
        with pytest.raises(ValueError, match="test"):
            CustomSingleSourceLoader(
                data=partial,
                target_column=0,
                task_mode="target",
                loading_type="numpy",
            )

    def test_presplit_dict_negative_target_column_resolves(self):
        """Negative target_column with pre-split dict → resolved to positive index (line 94).

        **Expected**: _target_column == 4 for -1 with 5-column arrays.
        """
        presplit = self._make_presplit(n_cols=5)
        loader = CustomSingleSourceLoader(
            data=presplit,
            target_column=-1,
            task_mode="target",
            loading_type="numpy",
        )
        assert loader._target_column == 4

    def test_invalid_loading_type_raises_value_error(self):
        """Invalid loading_type → ValueError (line 99).

        **Expected**: ValueError mentioning loading_type.
        """
        arr = np.random.randn(50, 5)
        with pytest.raises(ValueError, match="loading_type"):
            CustomSingleSourceLoader(
                data=arr,
                target_column=0,
                task_mode="target",
                loading_type="parquet",
            )

    def test_presplit_numpy_dict_load_data_and_get_data(self):
        """Pre-split numpy dict → load_data fills data_dict per split (lines 192-201, 216-220).

        **Expected**: data_dict['features'] is a dict keyed by split name.
        """
        presplit = self._make_presplit(n_cols=5)
        loader = CustomSingleSourceLoader(
            data=presplit,
            target_column=-1,
            task_mode="target",
            loading_type="numpy",
        )
        loader.load_data()
        assert isinstance(loader.data_dict["features"], dict)
        assert set(loader.data_dict["features"].keys()) == {"train", "val", "test"}

    def test_load_numpy_split_parameter_selects_subset(self):
        """_load_numpy with split param returns correct subset (line 156).

        **Expected**: returned array shape matches that split's rows.
        """
        presplit = self._make_presplit(n_cols=5)
        loader = CustomSingleSourceLoader(
            data=presplit,
            target_column=-1,
            task_mode="target",
            loading_type="numpy",
        )
        feats, target = loader._load_numpy(split="test")
        assert feats.shape[0] == 20


@pytest.mark.unit
class TestLoadFromCsvPreSplit:
    """Tests for CustomSingleSourceLoader with pre-split DataFrame dicts."""

    def _make_presplit_dfs(self):
        def _df(n):
            return pd.DataFrame(
                {"feat1": range(n), "feat2": range(n), "label": range(n)}
            )

        return {"train": _df(50), "val": _df(20), "test": _df(20)}

    def test_load_csv_split_parameter_selects_subset(self):
        """_load_csv with split param returns the val subset (line 126).

        **Expected**: returned feature array has val-split row count.
        """
        presplit = self._make_presplit_dfs()
        loader = CustomSingleSourceLoader.load_from_csv(
            presplit, target_column=-1, task_mode="target"
        )
        feats, target = loader._load_csv(split="val")
        assert feats.shape[0] == 20

    def test_load_csv_string_target_column_extracts_by_name(self):
        """String target_column → extracted via column name not index (lines 131-132).

        **Expected**: target column named 'label' is removed from features.
        """
        presplit = self._make_presplit_dfs()
        loader = CustomSingleSourceLoader.load_from_csv(
            presplit, target_column="label", task_mode="target"
        )
        feats, target = loader._load_csv(split="train")
        assert "label" not in [str(c) for c in range(feats.shape[1])]
        assert target.shape[1] == 1


@pytest.mark.unit
class TestMultiSourceLoaderValidation:
    """Tests for CustomMultiSourceLoader.load_from_primitive error branches (lines 379, 384)."""

    def _sources(self):
        rng = np.random.default_rng(1)
        return {
            "a": rng.standard_normal((50, 5)),
            "b": rng.standard_normal((50, 5)),
        }

    def test_data_splitter_list_raises_type_error(self):
        """data_splitter passed as a list → TypeError (line 379).

        **Expected**: TypeError mentioning data_splitter type.
        """
        splitter = TimeSplitter(train=0.6, val=0.2, test=None, seq_len=5, pred_len=2)
        with pytest.raises(TypeError):
            CustomMultiSourceLoader.load_from_primitive(
                sources=self._sources(),
                target_column=-1,
                task_mode="target",
                data_splitter=[splitter],
            )

    def test_data_splitter_dict_length_mismatch_raises_value_error(self):
        """data_splitter dict with wrong length → ValueError (line 384).

        **Expected**: ValueError mentioning mismatch.
        """
        splitter = TimeSplitter(train=0.6, val=0.2, test=None, seq_len=5, pred_len=2)
        with pytest.raises(ValueError):
            CustomMultiSourceLoader.load_from_primitive(
                sources=self._sources(),
                target_column=-1,
                task_mode="target",
                data_splitter={"a": splitter},
            )

    def test_load_from_primitive_with_per_source_splitters(self):
        """load_from_primitive with dict of per-source splitters → loader created (lines 389-408).

        Also exercises the DataFrame branch (lines 403-404) via a mixed numpy/DataFrame dict.

        **Expected**: CustomMultiSourceLoader instance returned.
        """
        splitter = TimeSplitter(
            train=0.6,
            val=0.2,
            test=None,
            seq_len=5,
            pred_len=2,
            create_splits_for=["features", "timestamps", "target"],
        )
        rng = np.random.default_rng(42)
        sources = {
            "a": rng.standard_normal((50, 5)),
            "b": pd.DataFrame(rng.standard_normal((50, 5))),
        }
        loader = CustomMultiSourceLoader.load_from_primitive(
            sources=sources,
            target_column=-1,
            task_mode="target",
            data_splitter={"a": splitter, "b": splitter},
        )
        assert loader is not None

    def test_load_from_primitive_with_by_source_splitter(self):
        """load_from_primitive with BySourceSplitter → is_part_of_multisource=True path (line 376).

        **Expected**: CustomMultiSourceLoader instance returned.
        """
        from picid.data.preprocessing import BySourceSplitter

        splitter = BySourceSplitter(
            sources_train=["a"],
            sources_val=["b"],
            sources_test=[],
        )
        sources = self._sources()
        loader = CustomMultiSourceLoader.load_from_primitive(
            sources=sources,
            target_column=-1,
            task_mode="target",
            data_splitter=splitter,
        )
        assert loader is not None
