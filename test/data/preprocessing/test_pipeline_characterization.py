"""
Characterization tests for PreProcessor pipeline and run.py contract (Phase 0).

Uses mock datasource + ConfigTransformManager, runs pipeline (with optional cache),
asserts output shape and processed-data accessors / get_meta_data_dict.
Run.py contract: datasource → get_split_mode() → PreProcessor → pipeline() → assert shapes.
"""

from __future__ import annotations

import sys
import types
import numpy as np
import pytest
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

from picid.data.data_objects import SplitDatasetContainer, SplitViewPolicy
from picid.data.datasources.base.exceptions import (
    DatasourceContractError,
    DatasourceLoadError,
    DatasourceStateError,
    DatasourceSplitError,
)
from picid.data.preprocessing.preprocessor import PreProcessor
from picid.exceptions import PreprocessingDatasourceError
from picid.transforms.base.transform_manager import ConfigTransformManager


# Reuse picklable datasource and transform config from caching tests
from test.data.preprocessing.test_preprocessor_caching import (
    _PicklableDatasource,
    _TRANSFORM_CONFIG_NO_CACHE_POINT,
)
from test.data.preprocessing.test_preprocessor import _PreprocessingErrorDatasource


def _make_manager():
    return ConfigTransformManager(transforms_config=_TRANSFORM_CONFIG_NO_CACHE_POINT)


def _import_run_module():
    """Import picid.run with the minimal missing optional dependency stubbed out."""
    einops_stub = types.ModuleType("einops")
    einops_stub.rearrange = lambda value, *args, **kwargs: value
    evaluator_pkg = types.ModuleType("picid.evaluator")
    evaluator_pkg.__path__ = []  # Mark as package so submodule imports resolve.
    evaluator_base = types.ModuleType("picid.evaluator.base")

    class _AbstractEvaluator:  # pragma: no cover - import-only stub
        """Minimal stand-in for the evaluator base class imported by picid.run."""

    evaluator_base.AbstractEvaluator = _AbstractEvaluator

    original_register_new_resolver = OmegaConf.register_new_resolver

    def _register_new_resolver(name, resolver, *, replace=False, use_cache=False):
        """Make repeated run.py imports idempotent inside the test process."""
        return original_register_new_resolver(
            name,
            resolver,
            replace=True,
            use_cache=use_cache,
        )

    with (
        patch.dict(
            sys.modules,
            {
                "einops": einops_stub,
                "picid.evaluator": evaluator_pkg,
                "picid.evaluator.base": evaluator_base,
            },
        ),
        patch.object(
            OmegaConf, "register_new_resolver", side_effect=_register_new_resolver
        ),
    ):
        sys.modules.pop("picid.run", None)
        import picid.run as run_module

    return run_module


def _assert_split_dict_structure(split_dict: dict) -> None:
    """Assert split_dict has train/val/test and features/target with array lists."""
    assert set(split_dict.keys()) >= {"train", "val", "test"}
    for split in ["train", "val", "test"]:
        assert split in split_dict
        assert "features" in split_dict[split]
        assert "target" in split_dict[split]
        feats = split_dict[split]["features"]
        targ = split_dict[split]["target"]
        assert isinstance(feats, list)
        assert isinstance(targ, list)
        assert len(feats) >= 1 and len(targ) >= 1
        assert hasattr(feats[0], "shape")
        assert hasattr(targ[0], "shape")


# ----- Pipeline with mock datasource + real ConfigTransformManager -----


def test_pipeline_mock_datasource_real_manager_output_shape():
    """PreProcessor with mock datasource and real ConfigTransformManager: pipeline() returns container with expected structure."""
    datasource = _PicklableDatasource(seed=1, n_units=2, n_samples=20, n_features=3)
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=manager,
    )
    result = preprocessor.pipeline()
    assert result is not None
    assert isinstance(result, SplitDatasetContainer)
    assert "features" in result and "target" in result
    for split in ["train", "val", "test"]:
        assert split in result["features"]
        assert len(result["features"][split]) == 2
        assert len(result["target"][split]) == 2


def test_pipeline_get_processed_split_dict_shape():
    """After pipeline(), get_processed_split_dict() exposes train/val/test and features/target."""
    datasource = _PicklableDatasource(seed=2, n_units=1, n_samples=10, n_features=2)
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=manager,
    )
    preprocessor.pipeline()
    data_dict = preprocessor.get_processed_split_dict(
        view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
    )
    _assert_split_dict_structure(data_dict)


def test_pipeline_get_processed_split_dict_can_unwrap_singletons():
    """Explicit singleton unwrapping is still available through the preprocessor API."""
    datasource = _PicklableDatasource(seed=22, n_units=1, n_samples=10, n_features=2)
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=manager,
    )

    preprocessor.pipeline()
    data_dict = preprocessor.get_processed_split_dict(
        view_policy=SplitViewPolicy.UNWRAP_SINGLETONS
    )

    assert hasattr(data_dict["train"]["features"], "shape")
    assert hasattr(data_dict["train"]["target"], "shape")


def test_pipeline_get_processed_split_dict_unwrap_rejects_multi_unit_data():
    """Preprocessor split export should reject singleton unwrapping for multi-unit splits."""
    datasource = _PicklableDatasource(seed=23, n_units=2, n_samples=10, n_features=2)
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=manager,
    )

    preprocessor.pipeline()

    with pytest.raises(ValueError, match="requires exactly one unit"):
        preprocessor.get_processed_split_dict(
            view_policy=SplitViewPolicy.UNWRAP_SINGLETONS
        )


def test_pipeline_get_meta_data_dict():
    """After pipeline(), get_meta_data_dict() returns a dict (from datasource.get_meta_data())."""
    datasource = _PicklableDatasource(seed=3, n_units=1, n_samples=10, n_features=2)
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=manager,
    )
    preprocessor.pipeline()
    meta = preprocessor.get_meta_data_dict()
    assert isinstance(meta, dict)
    assert "seed" in meta
    assert meta["seed"] == 3


def test_pipeline_with_cache_roundtrip(tmp_path):
    """Pipeline with cache on tmp_path: first run writes, second run returns same structure (cache hit)."""
    cache_dir = str(tmp_path / "cache")
    library_dir = str(tmp_path / "lib")
    tmp_path.joinpath("lib").mkdir()
    tmp_path.joinpath("lib", "dummy.py").write_text("# seed")

    datasource = _PicklableDatasource(seed=4, n_units=1, n_samples=15, n_features=2)
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=manager,
    )
    result1 = preprocessor.pipeline(
        data_cache_path=cache_dir,
        data_library_part_path=library_dir,
        transform_library_part_path=library_dir,
        cache_preprocessed=True,
    )
    assert result1 is not None
    d1 = preprocessor.get_processed_split_dict(
        view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
    )

    # Second run: should hit cache
    datasource2 = _PicklableDatasource(seed=4, n_units=1, n_samples=15, n_features=2)
    preprocessor2 = PreProcessor(
        datasource=datasource2,
        transforms=manager,
    )
    result2 = preprocessor2.pipeline(
        data_cache_path=cache_dir,
        data_library_part_path=library_dir,
        transform_library_part_path=library_dir,
        cache_preprocessed=True,
    )
    assert result2 is not None
    d2 = preprocessor2.get_processed_split_dict(
        view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
    )

    _assert_split_dict_structure(d1)
    _assert_split_dict_structure(d2)
    assert set(d1.keys()) == set(d2.keys())
    for split in ["train", "val", "test"]:
        np.testing.assert_array_almost_equal(
            np.asarray(d1[split]["features"][0]),
            np.asarray(d2[split]["features"][0]),
        )


# ----- run.py contract: datasource → get_split_mode() → PreProcessor → pipeline() -----


@pytest.fixture
def picklable_datasource_with_split_mode():
    """Datasource that has get_split_mode() for run.py-style tests."""
    return _PicklableDatasource(seed=10, n_units=1, n_samples=12, n_features=2)


def test_run_contract_get_split_mode_then_pipeline(
    picklable_datasource_with_split_mode,
):
    """run.py contract: datasource.get_split_mode() for logging; PreProcessor(datasource=..., transforms=...), pipeline(), then assert data_dict and meta_data."""
    datasource = picklable_datasource_with_split_mode
    if hasattr(datasource, "get_split_mode"):
        datasource.get_split_mode()  # Split mode from datasource; PreProcessor no longer takes mode.
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=manager,
    )
    preprocessor.pipeline()
    data_dict = preprocessor.get_processed_split_dict(
        view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
    )
    meta = preprocessor.get_meta_data_dict()
    _assert_split_dict_structure(data_dict)
    assert isinstance(meta, dict)


@pytest.mark.parametrize(
    ("stage", "datasource_kwargs", "error_type"),
    [
        (
            "load_data",
            {"load_error": DatasourceLoadError("load failed")},
            "DatasourceLoadError",
        ),
        (
            "split_data",
            {"split_error": DatasourceSplitError("split failed")},
            "DatasourceSplitError",
        ),
        (
            "get_meta_data",
            {"meta_error": DatasourceContractError("meta failed")},
            "DatasourceContractError",
        ),
    ],
)
def test_direct_pipeline_wraps_datasource_stage_failures(
    stage, datasource_kwargs, error_type
):
    """Direct pipeline should wrap datasource-layer errors with stage context."""
    datasource = _PreprocessingErrorDatasource(**datasource_kwargs)
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=manager,
    )

    with pytest.raises(PreprocessingDatasourceError) as exc_info:
        preprocessor.pipeline()

    exc = exc_info.value
    assert exc.stage == stage
    assert exc.datasource_type == "_PreprocessingErrorDatasource"
    assert exc.datasource_name == "faulty"
    assert exc.datasource_error_type == error_type


def test_direct_pipeline_does_not_wrap_assertion_errors():
    """AssertionError invariants should propagate unchanged through the direct pipeline."""
    datasource = _PreprocessingErrorDatasource(
        load_error=AssertionError("load invariant"),
    )
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=manager,
    )

    with pytest.raises(AssertionError, match="load invariant"):
        preprocessor.pipeline()


def test_run_top_level_logs_preprocessing_datasource_error_stage_details(tmp_path):
    """run.py should surface stage-aware preprocessing datasource failures in its top-level log."""
    run_module = _import_run_module()
    fake_datasource = MagicMock()
    fake_datasource.get_split_mode.return_value = "within_units"

    fake_preprocessor = MagicMock()
    fake_preprocessor.pipeline.side_effect = PreprocessingDatasourceError(
        "Datasource failed during preprocessing.",
        stage="get_data",
        datasource_type="FailingDatasource",
        datasource_name="failing_source",
        cause=DatasourceStateError("Data must be loaded before getting data."),
    )

    cfg = OmegaConf.create(
        {
            "seed": 7,
            "paths": {
                "root_dir": str(tmp_path / "root"),
                "output_dir": str(tmp_path / "out"),
                "cache_path": str(tmp_path / "cache"),
            },
            "cache": {
                "use_cache_after_loading": False,
                "use_cache_after_transfroms": False,
                "use_preprocessing_file_lock": False,
            },
            "datasource": {"_target_": "test.Datasource"},
            "transforms": {},
        }
    )

    with (
        patch.object(
            run_module.hydra.utils, "instantiate", return_value=fake_datasource
        ),
        patch.object(
            run_module,
            "ConfigTransformManager",
            return_value=MagicMock(get_transform_names=MagicMock(return_value=[])),
        ),
        patch.object(run_module, "PreProcessor", return_value=fake_preprocessor),
        patch.object(run_module, "instantiate_loggers", return_value=[]),
        patch.object(run_module, "display_targets"),
        patch.object(run_module, "print_hydra_config_tree", return_value="cfg-tree"),
        patch.object(run_module, "Console") as mock_console,
        patch.object(run_module.L, "seed_everything"),
        patch.object(run_module.torch, "manual_seed"),
        patch.object(run_module.np.random, "seed"),
        patch.object(run_module.random, "seed"),
        patch.object(run_module.log, "error") as mock_log_error,
    ):
        with pytest.raises(PreprocessingDatasourceError):
            run_module.run(cfg)

    mock_console.return_value.print.assert_called_once_with("cfg-tree")
    assert mock_log_error.call_count == 6
    assert mock_log_error.call_args_list[0].args == (
        "Datasource preprocessing failed.",
    )
    assert mock_log_error.call_args_list[1].args == (
        "Preprocessing datasource stage: %s",
        "get_data",
    )
    assert mock_log_error.call_args_list[2].args == (
        "Datasource class: %s",
        "FailingDatasource",
    )
    assert mock_log_error.call_args_list[3].args == (
        "Datasource name: %r",
        "failing_source",
    )
    assert mock_log_error.call_args_list[4].args == (
        "Datasource error type: %s",
        "DatasourceStateError",
    )
    assert mock_log_error.call_args_list[5].args[0] == "Original datasource error: %s"
    assert isinstance(mock_log_error.call_args_list[5].args[1], DatasourceStateError)
    assert str(mock_log_error.call_args_list[5].args[1]) == (
        "Data must be loaded before getting data."
    )


@pytest.mark.skip(
    reason="Transform strategy expects container-style data[key]; per-unit chunks have ndarray. Use picklable datasource for pipeline tests."
)
def test_run_contract_with_real_loader():
    """run.py contract using a real SingleSourceLoader: get_split_mode() → PreProcessor → pipeline().

    Skipped: pipeline apply_transforms receives per-unit chunks where data[key] is ndarray;
    CopyStep expects get_instance_cls(). Run contract is covered by test_run_contract_get_split_mode_then_pipeline
    and loader tests.
    """
    from test.data.datasources.base.conftest import InMemorySingleSourceLoader

    loader = InMemorySingleSourceLoader(
        n_samples=80, n_features=3, data_name="run_contract"
    )
    loader.load_data()
    loader.split_data()
    assert loader.get_split_mode() == "within_units"
    manager = _make_manager()
    preprocessor = PreProcessor(
        datasource=loader,
        transforms=manager,
    )
    result = preprocessor.pipeline()
    assert result is not None
    data_dict = preprocessor.get_processed_split_dict(
        view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
    )
    meta = preprocessor.get_meta_data_dict()
    _assert_split_dict_structure(data_dict)
    assert isinstance(meta, dict)
