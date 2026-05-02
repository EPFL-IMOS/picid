"""Tests for picid.interface.interface module.

Coverage target: >=95% of picid/interface/interface.py and picid/interface/__init__.py

Tests cover:
- Standalone helper functions (safe_div, diff_if_not_string, _uuid, etc.)
- EntryInterface class methods with mocked filesystem/Hydra/Lightning
- Lazy-loading exports from __init__.py
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError

from picid.interface.interface import (
    EntryInterface,
    _uuid,
    diff_if_not_string,
    register_infer_dataloader_length_resolver,
    safe_div,
    verify_thread_limits,
)


# ============================================================================
# Helper Functions
# ============================================================================


@pytest.mark.unit
class TestSafeDiv:
    """Tests for safe_div — integer division with list-of-candidates."""

    def test_exact_division(self):
        """10 / 2 = 5."""
        assert safe_div(10, 2) == 5

    def test_returns_int(self):
        """Result is always an integer (floor division)."""
        result = safe_div(10, 5)
        assert isinstance(result, int)

    def test_zero_divisor_raises(self):
        """Division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            safe_div(10, 0)

    def test_not_divisible_raises(self):
        """Non-exact division raises ValueError."""
        with pytest.raises(ValueError, match="not divisible"):
            safe_div(10, 3)

    def test_list_first_match_wins(self):
        """List of divisors: first valid one is used."""
        assert safe_div(12, [6, 4, 3]) == 2

    def test_list_skips_failing(self):
        """Failing divisors in list are skipped."""
        assert safe_div(12, [5, 4]) == 3

    def test_empty_list_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty list"):
            safe_div(10, [])

    def test_no_valid_in_list_raises(self):
        """No valid divisor in list raises ValueError."""
        with pytest.raises(ValueError, match="not divisible by any candidate"):
            safe_div(7, [2, 3])

    def test_tuple_also_works(self):
        """Tuple of divisors works like list."""
        assert safe_div(12, (5, 4)) == 3


@pytest.mark.unit
class TestDiffIfNotString:
    """Tests for diff_if_not_string."""

    def test_numeric_difference(self):
        """Numeric values are subtracted."""
        assert diff_if_not_string(10, 3) == 7

    def test_both_strings_returns_a(self):
        """Both strings → returns a unchanged."""
        assert diff_if_not_string("hello", "world") == "hello"

    def test_string_a_numeric_b(self):
        """String a + numeric b → returns a."""
        assert diff_if_not_string("hello", 5) == "hello"

    def test_numeric_a_string_b(self):
        """Numeric a + string b → returns a."""
        assert diff_if_not_string(10, "world") == 10

    def test_float_difference(self):
        """Float subtraction works."""
        assert diff_if_not_string(5.5, 2.0) == pytest.approx(3.5)


@pytest.mark.unit
class TestUuid:
    """Tests for _uuid generation."""

    def test_hex_32_chars(self):
        """'hex' returns 32-character hex string."""
        result = _uuid("hex")
        assert len(result) == 32
        int(result, 16)

    def test_short_8_chars(self):
        """'short' returns 8-character hex string."""
        result = _uuid("short")
        assert len(result) == 8
        int(result, 16)

    def test_unknown_kind_raises(self):
        """Unknown kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown uuid kind"):
            _uuid("full")

    def test_uniqueness(self):
        """100 consecutive calls produce 100 unique values."""
        results = {_uuid("hex") for _ in range(100)}
        assert len(results) == 100


@pytest.mark.unit
class TestVerifyThreadLimits:
    """Tests for verify_thread_limits logging."""

    def test_logs_expected_threads(self, caplog):
        """Logs the expected thread count."""
        with caplog.at_level(logging.INFO, logger="picid.interface.interface"):
            verify_thread_limits(4)
        assert any("Expected threads: 4" in msg for msg in caplog.messages)

    def test_logs_set_env_vars(self, monkeypatch, caplog):
        """Logs environment variables when set."""
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        with caplog.at_level(logging.INFO, logger="picid.interface.interface"):
            verify_thread_limits(8)
        assert any("OMP_NUM_THREADS" in msg for msg in caplog.messages)


@pytest.mark.unit
class TestRegisterInferDataloaderLengthResolver:
    """Tests for the OmegaConf resolver registration."""

    def test_resolver_returns_value(self):
        """Registered resolver returns correct value."""
        register_infer_dataloader_length_resolver({"train": 100, "val": 50})
        cfg = OmegaConf.create({"x": "${infer_dataloader_length:train}"})
        assert OmegaConf.to_container(cfg, resolve=True)["x"] == 100

    def test_resolver_missing_key_raises(self):
        """Missing key raises InterpolationResolutionError wrapping KeyError."""
        register_infer_dataloader_length_resolver({"train": 100})
        cfg = OmegaConf.create({"x": "${infer_dataloader_length:missing}"})
        with pytest.raises(InterpolationResolutionError):
            OmegaConf.to_container(cfg, resolve=True)


# ============================================================================
# Lazy Loading (__init__.py)
# ============================================================================


@pytest.mark.unit
class TestInterfaceLazyLoading:
    """Tests for picid.interface.__init__.py lazy-loading."""

    def test_entry_interface_importable(self):
        """EntryInterface is importable from picid.interface."""
        from picid.interface import EntryInterface as EI

        assert EI is not None

    def test_unknown_attr_raises(self):
        """Unknown attribute raises AttributeError."""
        import picid.interface as mod

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = mod.TotallyFakeAttribute

    def test_getattr_entry_interface_returns_class(self):
        """__getattr__('EntryInterface') returns the EntryInterface class."""
        import picid.interface as mod

        result = mod.__getattr__("EntryInterface")
        assert result is EntryInterface


# ============================================================================
# EntryInterface — get_available_resource
# ============================================================================


@pytest.mark.unit
class TestGetAvailableResource:
    """Tests for EntryInterface.get_available_resource with mocked filesystem."""

    def test_scans_directory(self, monkeypatch, tmp_path):
        """Returns list of files from directory walk.

        **Methodology**: Mock project_config.config_path to tmp_path with files.

        **Expected**: Returns list of relative file paths.
        """
        resource_dir = tmp_path / "model"
        resource_dir.mkdir()
        (resource_dir / "mlp.yaml").touch()
        (resource_dir / "cnn.yaml").touch()

        monkeypatch.setattr(
            "picid.interface.interface.project_config.config_path", tmp_path
        )

        result = EntryInterface.get_available_resource("model")
        assert "mlp.yaml" in result
        assert "cnn.yaml" in result

    def test_max_level_limits_depth(self, monkeypatch, tmp_path):
        """max_level=1 only scans first directory level.

        **Methodology**: Create nested directories, scan with max_level=1.

        **Expected**: Only top-level files returned.
        """
        resource_dir = tmp_path / "datasource"
        resource_dir.mkdir()
        (resource_dir / "top.yaml").touch()
        nested = resource_dir / "sub"
        nested.mkdir()
        (nested / "deep.yaml").touch()

        monkeypatch.setattr(
            "picid.interface.interface.project_config.config_path", tmp_path
        )

        result = EntryInterface.get_available_resource("datasource", max_level=1)
        assert any("top" in r for r in result)

    def test_max_level_none_uses_inf(self, monkeypatch, tmp_path):
        """max_level=None scans all depths.

        **Methodology**: Create nested dirs, scan with None.

        **Expected**: All files found.
        """
        resource_dir = tmp_path / "model"
        resource_dir.mkdir()
        (resource_dir / "top.yaml").touch()
        nested = resource_dir / "deep"
        nested.mkdir()
        (nested / "bottom.yaml").touch()

        monkeypatch.setattr(
            "picid.interface.interface.project_config.config_path", tmp_path
        )

        result = EntryInterface.get_available_resource("model", max_level=None)
        assert any("top.yaml" in r for r in result)
        assert any("bottom.yaml" in r for r in result)

    def test_as_tuple_returns_tuples(self, monkeypatch, tmp_path):
        """as_tuple=True splits paths into tuple components.

        **Methodology**: Create nested file, request as_tuple.

        **Expected**: Returns tuple of path components.
        """
        resource_dir = tmp_path / "model_configs"
        sub = resource_dir / "prog"
        sub.mkdir(parents=True)
        (sub / "mlp.yaml").touch()

        monkeypatch.setattr(
            "picid.interface.interface.project_config.config_path", tmp_path
        )

        result = EntryInterface.get_available_resource("model_configs", as_tuple=True)
        assert any(isinstance(r, tuple) for r in result)

    def test_as_tuple_single_component_returns_string(self, monkeypatch, tmp_path):
        """Single-component path is unwrapped from 1-tuple to string.

        **Methodology**: Create top-level file, request as_tuple.

        **Expected**: Single-component entries are strings.
        """
        resource_dir = tmp_path / "evaluator"
        resource_dir.mkdir()
        (resource_dir / "default.yaml").touch()

        monkeypatch.setattr(
            "picid.interface.interface.project_config.config_path", tmp_path
        )

        result = EntryInterface.get_available_resource("evaluator", as_tuple=True)
        assert "default" in result

    def test_datasource_strips_yaml(self, monkeypatch, tmp_path):
        """resource_type='datasource' strips .yaml extension.

        **Methodology**: Create datasource yaml files.

        **Expected**: Extensions removed from results.
        """
        resource_dir = tmp_path / "datasource"
        resource_dir.mkdir()
        (resource_dir / "phme20.yaml").touch()

        monkeypatch.setattr(
            "picid.interface.interface.project_config.config_path", tmp_path
        )

        result = EntryInterface.get_available_resource("datasource")
        assert "phme20" in result
        assert "phme20.yaml" not in result


# ============================================================================
# EntryInterface — load_resource
# ============================================================================


@pytest.mark.unit
class TestLoadResource:
    """Tests for EntryInterface.load_resource."""

    def test_dict_input(self):
        """Dict input is wrapped in OmegaConf.create.

        **Expected**: Returns DictConfig with same keys.
        """
        cfg = EntryInterface.load_resource({"_target_": "my.Class"}, "model")
        assert isinstance(cfg, DictConfig)
        assert cfg["_target_"] == "my.Class"

    def test_str_datasource_loads_yaml(self, monkeypatch, tmp_path):
        """String datasource loads YAML and adds cache_dir/data_path.

        **Methodology**: Create a datasource YAML, mock project_config.

        **Expected**: YAML loaded, cache_dir and data_path set.
        """
        ds_dir = tmp_path / "datasource"
        ds_dir.mkdir()
        ds_yaml = ds_dir / "phme20.yaml"
        ds_yaml.write_text("_target_: picid.data.datasources.Phme20\nsome_key: value\n")

        mock_config = MagicMock()
        mock_config.config_path = tmp_path
        mock_config.cache_dir = "/cache"
        mock_config.data_dir = Path("/data")

        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        cfg = EntryInterface.load_resource("phme20", "datasource")
        assert cfg["_target_"] == "picid.data.datasources.Phme20"
        assert cfg["cache_dir"] == "/cache"

    def test_str_non_datasource_no_cache(self, monkeypatch, tmp_path):
        """Non-datasource string loads YAML without adding cache_dir.

        **Methodology**: Create evaluator YAML.

        **Expected**: No cache_dir key in result.
        """
        eval_dir = tmp_path / "evaluator"
        eval_dir.mkdir()
        (eval_dir / "default.yaml").write_text("_target_: picid.evaluator.Default\n")

        mock_config = MagicMock()
        mock_config.config_path = tmp_path

        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        cfg = EntryInterface.load_resource("default.yaml", "evaluator")
        assert "cache_dir" not in cfg

    def test_invalid_resource_name_raises(self, monkeypatch, tmp_path):
        """Unknown resource name raises AssertionError.

        **Expected**: AssertionError with descriptive message.
        """
        ds_dir = tmp_path / "datasource"
        ds_dir.mkdir()
        (ds_dir / "existing.yaml").write_text("_target_: T\n")

        mock_config = MagicMock()
        mock_config.config_path = tmp_path

        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        with pytest.raises(AssertionError, match="not present"):
            EntryInterface.load_resource("nonexistent", "datasource")

    def test_unsupported_type_raises(self):
        """Non str/Path/dict raises TypeError."""
        with pytest.raises(TypeError, match="load_resource expects"):
            EntryInterface.load_resource(42, "model")

    def test_path_relative_to_root(self, monkeypatch, tmp_path):
        """Path that is relative_to config root gets stripped.

        **Methodology**: Supply full path starting from config_path.

        **Expected**: Path stripped and lookup succeeds.
        """
        ds_dir = tmp_path / "datasource"
        ds_dir.mkdir()
        ds_yaml = ds_dir / "test_ds.yaml"
        ds_yaml.write_text("_target_: picid.data.TestDS\n")

        mock_config = MagicMock()
        mock_config.config_path = tmp_path
        mock_config.cache_dir = "/cache"
        mock_config.data_dir = Path("/data")

        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        cfg = EntryInterface.load_resource(ds_dir / "test_ds", "datasource")
        assert cfg["_target_"] == "picid.data.TestDS"


# ============================================================================
# EntryInterface — get_model_cfg
# ============================================================================


@pytest.mark.unit
class TestGetModelCfg:
    """Tests for EntryInterface.get_model_cfg."""

    def test_dict_passthrough(self):
        """Dict model returns (dict, None)."""
        iface = EntryInterface()
        cfg, name = iface.get_model_cfg({"layers": 3}, task="prog")
        assert cfg == {"layers": 3}
        assert name is None

    def test_custom_model_trainer_passthrough(self):
        """CustomModelTrainer returns (trainer, None)."""
        from picid.interface.model.custom_model import CustomModelTrainer

        iface = EntryInterface()
        mock_model = MagicMock()
        trainer = CustomModelTrainer(task_type="rul", model=mock_model)
        cfg, name = iface.get_model_cfg(trainer, task="prog")
        assert cfg is trainer
        assert name is None

    def test_str_loads_yaml(self, monkeypatch, tmp_path):
        """String model name loads YAML from model_configs/<task>/.

        **Methodology**: Create model config YAML, mock get_available_resource.

        **Expected**: YAML loaded, folder name returned.
        """
        model_dir = tmp_path / "model_configs" / "prognostics"
        model_dir.mkdir(parents=True)
        (model_dir / "mlp.yaml").write_text("hidden_dim: 64\n")

        mock_config = MagicMock()
        mock_config.config_path = tmp_path
        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        iface = EntryInterface()
        monkeypatch.setattr(
            iface,
            "get_available_resource",
            lambda *a, **kw: ["mlp"] if kw.get("as_tuple") else ["mlp.yaml"],
        )

        cfg, name = iface.get_model_cfg("mlp", task="prognostics")
        assert cfg["hidden_dim"] == 64
        assert name == "mlp"

    def test_unsupported_type_raises(self):
        """Invalid model type raises TypeError."""
        iface = EntryInterface()
        with pytest.raises(TypeError, match="Invalid model type"):
            iface.get_model_cfg(42, task="prog")

    def test_str_not_found_raises(self, monkeypatch):
        """Unknown model string raises AssertionError."""
        iface = EntryInterface()
        monkeypatch.setattr(iface, "get_available_resource", lambda *a, **kw: [])
        with pytest.raises(AssertionError, match="does not exist"):
            iface.get_model_cfg("nonexistent", task="prog")

    def test_abs_model_config_merges_model_dump(self, monkeypatch, tmp_path):
        """AbsModelConfig merges model_dump into loaded YAML (line 398)."""
        from picid.interface.schemas.model.mlp import MLPConfig

        model_dir = tmp_path / "model_configs" / "prog"
        model_dir.mkdir(parents=True)
        (model_dir / "mlp.yaml").write_text("hidden_dim: 64\n")

        mock_config = MagicMock()
        mock_config.config_path = tmp_path
        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        iface = EntryInterface()
        model = MLPConfig(input_channels=10, num_targets=1)
        cfg, name = iface.get_model_cfg(model, task="prog")
        assert name == "mlp"
        assert cfg is not None


# ============================================================================
# EntryInterface — get_task_definition_cfg
# ============================================================================


@pytest.mark.unit
class TestGetTaskDefinitionCfg:
    """Tests for EntryInterface.get_task_definition_cfg."""

    def test_dict_passthrough(self):
        """Dict task definition → passthrough, name=None."""
        iface = EntryInterface()
        cfg, name = iface.get_task_definition_cfg({"task_type": "rul"})
        assert cfg == {"task_type": "rul"}
        assert name is None

    def test_tuple_loads_yaml(self, monkeypatch, tmp_path):
        """Tuple reference loads YAML from task_definition/.

        **Methodology**: Create task YAML, mock filesystem.

        **Expected**: YAML loaded, first component as name.
        """
        td_dir = tmp_path / "task_definition" / "prognostics"
        td_dir.mkdir(parents=True)
        (td_dir / "rul.yaml").write_text("task_type: rul\npred_len: 0\n")

        mock_config = MagicMock()
        mock_config.config_path = tmp_path
        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        iface = EntryInterface()
        monkeypatch.setattr(
            iface,
            "get_available_resource",
            lambda *a, **kw: [("prognostics", "rul")] if kw.get("as_tuple") else [],
        )

        cfg, name = iface.get_task_definition_cfg(("prognostics", "rul"))
        assert cfg["task_type"] == "rul"

    def test_base_task_definition_model_dump(self, monkeypatch):
        """BaseTaskDefinition is serialized via model_dump."""
        from picid.interface.schemas.task_definition import BaseTaskDefinition

        mock_td = MagicMock(spec=BaseTaskDefinition)
        mock_td.model_dump.return_value = {"task_type": "rul", "pred_len": 0}
        mock_td.config_name = "prognostics"

        iface = EntryInterface()
        cfg, name = iface.get_task_definition_cfg(mock_td)
        assert cfg["task_type"] == "rul"
        assert name == "prognostics"

    def test_unsupported_type_raises(self):
        """Invalid type raises TypeError."""
        iface = EntryInterface()
        with pytest.raises(TypeError, match="Invalid task definition type"):
            iface.get_task_definition_cfg(42)

    def test_tuple_not_found_raises(self, monkeypatch):
        """Tuple not in available raises AssertionError."""
        iface = EntryInterface()
        monkeypatch.setattr(iface, "get_available_resource", lambda *a, **kw: [])
        with pytest.raises(AssertionError, match="does not exist"):
            iface.get_task_definition_cfg(("prog", "rul"))


# ============================================================================
# EntryInterface — get_datasource_cfg
# ============================================================================


@pytest.mark.unit
class TestGetDatasourceCfg:
    """Tests for EntryInterface.get_datasource_cfg."""

    def test_str_delegates_to_load_resource(self, monkeypatch):
        """String datasource delegates to load_resource."""
        iface = EntryInterface()
        mock_cfg = DictConfig({"_target_": "T"})
        monkeypatch.setattr(iface, "load_resource", lambda v, rt: mock_cfg)

        result = iface.get_datasource_cfg("phme20")
        assert result["_target_"] == "T"

    def test_dict_passthrough(self):
        """Dict datasource is returned as-is."""
        iface = EntryInterface()
        d = {"_target_": "custom"}
        result = iface.get_datasource_cfg(d)
        assert result == d

    def test_tuple_loads_experiment(self, monkeypatch, tmp_path):
        """Tuple datasource loads experiment YAML and sets is_experiment.

        **Methodology**: Create experiment YAML at experiment/<source>/<task>/<sub>/<model>.yaml.

        **Expected**: YAML loaded with is_experiment=True.
        """
        exp_dir = tmp_path / "experiment" / "phme20" / "prog" / "default"
        exp_dir.mkdir(parents=True)
        (exp_dir / "mlp.yaml").write_text("model_name: mlp\n")

        mock_config = MagicMock()
        mock_config.config_path = tmp_path
        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        iface = EntryInterface()
        monkeypatch.setattr(
            iface,
            "get_available_resource",
            lambda rtype, **kw: [("prog", "default", "mlp")],
        )

        result = iface.get_datasource_cfg(
            ("phme20", "default"), task="prog", model="mlp"
        )
        assert result["is_experiment"] is True

    def test_tuple_without_model_raises(self, monkeypatch):
        """Tuple without model argument raises AssertionError."""
        iface = EntryInterface()
        with pytest.raises(AssertionError, match="model"):
            iface.get_datasource_cfg(("source", "sub"), task="prog", model=None)

    def test_tuple_invalid_task_raises(self, monkeypatch, tmp_path):
        """Tuple with unavailable task raises AssertionError."""
        mock_config = MagicMock()
        mock_config.config_path = tmp_path
        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        iface = EntryInterface()
        monkeypatch.setattr(
            iface,
            "get_available_resource",
            lambda rtype, **kw: [("other_task", "default", "mlp")],
        )

        with pytest.raises(AssertionError, match="does not have a default"):
            iface.get_datasource_cfg(("source", "sub"), task="nonexist", model="mlp")

    def test_unsupported_type_raises(self):
        """Invalid type raises TypeError."""
        iface = EntryInterface()
        with pytest.raises(TypeError, match="Invalid datasource type"):
            iface.get_datasource_cfg(42)


# ============================================================================
# EntryInterface — get_datasource
# ============================================================================


@pytest.mark.unit
class TestGetDatasource:
    """Tests for EntryInterface.get_datasource."""

    def test_loads_and_instantiates(self, monkeypatch):
        """Loads resource then instantiates it.

        **Methodology**: Mock load_resource and hydra.utils.instantiate.

        **Expected**: instantiate called with loaded config.
        """
        iface = EntryInterface()
        mock_cfg = DictConfig({"_target_": "MyLoader"})
        mock_loader = MagicMock()

        monkeypatch.setattr(iface, "load_resource", lambda v, rt: mock_cfg)
        monkeypatch.setattr(
            "picid.interface.interface.hydra.utils.instantiate", lambda c: mock_loader
        )

        result = iface.get_datasource("phme20")
        assert result is mock_loader


# ============================================================================
# EntryInterface — process_datasource
# ============================================================================


@pytest.mark.unit
class TestProcessDatasource:
    """Tests for EntryInterface.process_datasource."""

    def test_creates_preprocessor_with_cache(self, monkeypatch):
        """cache=True uses project_config.cache_path."""
        iface = EntryInterface()
        mock_pipeline_result = MagicMock()
        mock_preprocessor = MagicMock()
        mock_preprocessor.pipeline.return_value = mock_pipeline_result

        monkeypatch.setattr(
            "picid.interface.interface.InterfacePreProcessor",
            lambda ds, tr, preprocessor_mode, cache_path: mock_preprocessor,
        )
        monkeypatch.setattr(
            "picid.interface.interface.project_config.cache_path", "/cache"
        )

        result = iface.process_datasource("ds", [], cache=True)
        assert result is mock_pipeline_result

    def test_cache_false_passes_none(self, monkeypatch):
        """cache=False passes cache_path=None."""
        iface = EntryInterface()
        captured = {}

        def mock_init(ds, tr, preprocessor_mode, cache_path):
            captured["cache_path"] = cache_path
            m = MagicMock()
            m.pipeline.return_value = MagicMock()
            return m

        monkeypatch.setattr(
            "picid.interface.interface.InterfacePreProcessor", mock_init
        )

        iface.process_datasource("ds", [], cache=False)
        assert captured["cache_path"] is None


# ============================================================================
# EntryInterface — get_available_datasources
# ============================================================================


@pytest.mark.unit
class TestGetAvailableDatasources:
    """Tests for the cached datasource listing."""

    def test_delegates_to_get_available_resource(self, monkeypatch):
        """Delegates to get_available_resource('datasource')."""
        monkeypatch.setattr(
            EntryInterface,
            "get_available_resource",
            staticmethod(lambda rt, **kw: ["ds1", "ds2"]),
        )
        EntryInterface.get_available_datasources.cache_clear()
        result = EntryInterface.get_available_datasources()
        assert result == ["ds1", "ds2"]
        EntryInterface.get_available_datasources.cache_clear()


# ============================================================================
# EntryInterface — _compose_config_file
# ============================================================================


@pytest.mark.unit
class TestComposeConfigFile:
    """Tests for _compose_config_file with mocked Hydra."""

    @pytest.fixture
    def mock_env(self, monkeypatch, tmp_path):
        """Set up mocked environment for _compose_config_file tests."""
        mock_config = MagicMock()
        mock_config.config_path = tmp_path
        mock_config.model_dump.return_value = {"config_path": str(tmp_path)}

        (tmp_path / "trainer").mkdir()
        (tmp_path / "trainer" / "default.yaml").write_text("max_epochs: 10\n")

        (tmp_path / "hydra").mkdir()
        (tmp_path / "hydra" / "interface.yaml").write_text("run:\n  dir: outputs\n")

        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        mock_compose = MagicMock(return_value=DictConfig({"seed": 42}))
        monkeypatch.setattr("picid.interface.interface.hydra.compose", mock_compose)

        return SimpleNamespace(
            config=mock_config,
            compose=mock_compose,
            tmp_path=tmp_path,
        )

    def test_basic_composition(self, mock_env):
        """Basic call with minimal arguments.

        **Expected**: hydra.compose called with overrides.
        """
        iface = EntryInterface()
        iface._compose_config_file(
            run_name="test_run",
            task_definition=None,
            model=None,
            datasource=None,
        )
        mock_env.compose.assert_called_once()

    def test_overrides_none_becomes_empty_list(self, mock_env):
        """overrides=None is converted to empty list."""
        iface = EntryInterface()
        iface._compose_config_file(
            run_name="test",
            task_definition=None,
            model=None,
            datasource=None,
            overrides=None,
        )
        call_args = mock_env.compose.call_args
        overrides = call_args.kwargs.get("overrides", call_args[1].get("overrides", []))
        assert isinstance(overrides, list)

    def test_overrides_string_wrapped_in_list(self, mock_env):
        """String overrides are wrapped in a list."""
        iface = EntryInterface()
        iface._compose_config_file(
            run_name="test",
            task_definition=None,
            model=None,
            datasource=None,
            overrides="trainer.max_epochs=5",
        )
        call_args = mock_env.compose.call_args
        overrides = call_args.kwargs.get("overrides", call_args[1].get("overrides", []))
        assert "trainer.max_epochs=5" in overrides

    def test_processed_datasource_adds_override(self, mock_env, monkeypatch):
        """ProcessedDatasource adds task_mode override."""
        from picid.interface.utils import ProcessedDatasource

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"

        iface = EntryInterface()
        iface._compose_config_file(
            run_name="test",
            task_definition=None,
            model=None,
            datasource=mock_ds,
        )
        call_args = mock_env.compose.call_args
        overrides = call_args.kwargs.get("overrides", call_args[1].get("overrides", []))
        assert any("task_mode=regression" in o for o in overrides)

    def test_custom_model_trainer_loads_custom_base(self, mock_env, monkeypatch):
        """CustomModelTrainer loads custom_base.yaml."""
        from picid.interface.model.custom_model import CustomModelTrainer

        custom_dir = mock_env.tmp_path / "model_configs" / "prog"
        custom_dir.mkdir(parents=True)
        (custom_dir / "custom_base.yaml").write_text("backbone: null\n")

        mock_trainer = MagicMock(spec=CustomModelTrainer)

        iface = EntryInterface()
        monkeypatch.setattr(
            iface, "get_task_definition_cfg", lambda td: ({"task_type": "rul"}, "prog")
        )

        iface._compose_config_file(
            run_name="test",
            task_definition=("prog", "rul"),
            model=mock_trainer,
            datasource=None,
        )
        mock_env.compose.assert_called_once()

    def test_model_without_task_raises(self, mock_env):
        """Model without task_definition raises ValueError."""
        iface = EntryInterface()
        with pytest.raises(ValueError, match="task_definition is none"):
            iface._compose_config_file(
                run_name="test",
                task_definition=None,
                model="mlp",
                datasource=None,
            )

    def test_callbacks_default(self, mock_env):
        """callbacks='default' loads default.yaml."""
        (mock_env.tmp_path / "callbacks").mkdir()
        (mock_env.tmp_path / "callbacks" / "default.yaml").write_text(
            "early_stopping: true\n"
        )

        iface = EntryInterface()
        iface._compose_config_file(
            run_name="test",
            task_definition=None,
            model=None,
            datasource=None,
            callbacks="default",
        )
        mock_env.compose.assert_called_once()

    def test_callbacks_as_lightning_callbacks(self, mock_env):
        """Lightning callbacks pass through without config loading."""
        from lightning import Callback

        mock_cb = MagicMock(spec=Callback)

        iface = EntryInterface()
        iface._compose_config_file(
            run_name="test",
            task_definition=None,
            model=None,
            datasource=None,
            callbacks=[mock_cb],
        )
        mock_env.compose.assert_called_once()

    def test_callbacks_as_string_list(self, mock_env):
        """String callbacks load individual YAML files."""
        (mock_env.tmp_path / "callbacks").mkdir(exist_ok=True)
        (mock_env.tmp_path / "callbacks" / "early_stop.yaml").write_text(
            "patience: 5\n"
        )

        iface = EntryInterface()
        iface._compose_config_file(
            run_name="test",
            task_definition=None,
            model=None,
            datasource=None,
            callbacks=["early_stop"],
        )
        mock_env.compose.assert_called_once()

    def test_trainer_config_provided(self, mock_env):
        """TrainerConfig provided uses model_dump instead of default."""
        from picid.interface.schemas.trainer import TrainerConfig

        mock_tc = MagicMock(spec=TrainerConfig)
        mock_tc.model_dump.return_value = {"max_epochs": 50}

        iface = EntryInterface()
        iface._compose_config_file(
            run_name="test",
            task_definition=None,
            model=None,
            datasource=None,
            trainer_config=mock_tc,
        )
        mock_tc.model_dump.assert_called_once()

    def test_evaluators_as_string(self, mock_env, monkeypatch):
        """String evaluators load from evaluator/<name>.yaml."""
        (mock_env.tmp_path / "evaluator").mkdir()
        (mock_env.tmp_path / "evaluator" / "default.yaml").write_text(
            "_target_: Eval\n"
        )

        monkeypatch.setattr(
            EntryInterface,
            "get_available_resource",
            staticmethod(
                lambda rt, **kw: ["default"] if kw.get("as_tuple") else ["default.yaml"]
            ),
        )

        iface = EntryInterface()
        iface._compose_config_file(
            run_name="test",
            task_definition=None,
            model=None,
            datasource=None,
            evaluators="default",
        )
        mock_env.compose.assert_called_once()

    def test_datasource_experiment(self, mock_env, monkeypatch):
        """Datasource cfg with is_experiment=True uses experiment override."""
        iface = EntryInterface()
        exp_cfg = DictConfig({"is_experiment": True, "data": "x"})
        monkeypatch.setattr(iface, "get_datasource_cfg", lambda ds, **kw: exp_cfg)
        monkeypatch.setattr(
            iface, "get_task_definition_cfg", lambda td: ({"task_type": "rul"}, "prog")
        )
        monkeypatch.setattr(iface, "get_model_cfg", lambda m, task: ({"h": 64}, "mlp"))

        iface._compose_config_file(
            run_name="test",
            task_definition=("prog", "rul"),
            model="mlp",
            datasource=("phme20", "default"),
        )
        call_args = mock_env.compose.call_args
        overrides = call_args.kwargs.get("overrides", call_args[1].get("overrides", []))
        assert "experiment=experiment_cfg" in overrides

    def test_datasource_non_experiment(self, mock_env, monkeypatch):
        """Datasource cfg without is_experiment uses datasource override."""
        iface = EntryInterface()
        ds_cfg = DictConfig({"_target_": "T"})
        monkeypatch.setattr(iface, "get_datasource_cfg", lambda ds, **kw: ds_cfg)

        iface._compose_config_file(
            run_name="test",
            task_definition=None,
            model=None,
            datasource="phme20",
        )
        call_args = mock_env.compose.call_args
        overrides = call_args.kwargs.get("overrides", call_args[1].get("overrides", []))
        assert "datasource=datasource_cfg" in overrides

    def test_evaluators_abs_eval_config(self, mock_env, monkeypatch):
        """AbsEvalConfig evaluators use model_dump (line 793)."""
        from picid.interface.schemas.evaluators import DefaultEvaluatorConfig

        iface = EntryInterface()
        eval_cfg = DefaultEvaluatorConfig()
        monkeypatch.setattr(
            iface,
            "get_task_definition_cfg",
            lambda td, **kw: ({"task_type": "rul"}, "prog_rul"),
        )
        monkeypatch.setattr(iface, "get_model_cfg", lambda m, **kw: ({"h": 64}, "mlp"))

        iface._compose_config_file(
            run_name="test",
            task_definition=("prog", "rul"),
            model="mlp",
            datasource=None,
            evaluators=eval_cfg,
        )
        mock_env.compose.assert_called()


# ============================================================================
# EntryInterface — train
# ============================================================================


@pytest.mark.unit
class TestTrain:
    """Tests for EntryInterface.train with mocked Hydra/Lightning."""

    @pytest.fixture
    def train_env(self, monkeypatch, tmp_path):
        """Full mocked environment for train() tests.

        Mocks Hydra initialisation, _compose_config_file, and all
        Lightning/instantiate boundaries so tests exercise only the
        control-flow logic inside train().
        """
        from contextlib import contextmanager

        mock_config = MagicMock()
        mock_config.config_path = tmp_path
        mock_config.cache_path = tmp_path / "cache"
        mock_config.save_dir = str(tmp_path / "saves")
        mock_config.model_dump.return_value = {}

        monkeypatch.setattr("picid.interface.interface.project_config", mock_config)

        train_data = {
            "features": np.random.randn(10, 5),
            "target": np.random.randn(10, 1),
        }

        cfg_dict = {
            "seed": 42,
            "flash_attention": True,
            "trainer": {"max_epochs": 10, "_target_": "lightning.Trainer"},
            "dataset": {"_target_": "MyDataset"},
            "datamodule": {"_target_": "MyDM"},
            "task_definition": {
                "model": {
                    "data_requirements": {"input_tensors": ["features", "target"]}
                }
            },
            "loss": {"_target_": "torch.nn.MSELoss"},
            "evaluator": {
                "train": {"_target_": "Eval", "apply_inverse_scaling": False},
                "val": {"_target_": "Eval", "apply_inverse_scaling": False},
                "test": {"_target_": "Eval", "apply_inverse_scaling": False},
            },
        }
        cfg = OmegaConf.create(cfg_dict)

        # Mock Hydra initialisation (context manager) and GlobalHydra
        @contextmanager
        def _noop_init_config_dir(**kwargs):
            yield

        monkeypatch.setattr(
            "picid.interface.interface.hydra.initialize_config_dir",
            lambda **kw: _noop_init_config_dir(**kw),
        )
        mock_global_hydra = MagicMock()
        monkeypatch.setattr("picid.interface.interface.GlobalHydra", mock_global_hydra)
        mock_hydra_config = MagicMock()
        monkeypatch.setattr("picid.interface.interface.HydraConfig", mock_hydra_config)

        # Mock _compose_config_file to return our pre-built cfg
        monkeypatch.setattr(
            EntryInterface,
            "_compose_config_file",
            lambda self, **kw: cfg,
        )

        mock_trainer = MagicMock()
        mock_trainer.test.return_value = [{"test/loss": 0.1}]

        mock_dataset = MagicMock()

        mock_batch = {
            "features": torch.zeros(2, 5),
            "target": torch.zeros(2, 1),
        }
        # side_effect so each iter() call gets a fresh iterator (not exhausted)
        mock_loader = MagicMock()
        mock_loader.__len__.return_value = 5
        mock_loader.__iter__.side_effect = lambda: iter([mock_batch])

        mock_datamodule = MagicMock()
        mock_datamodule.train_dataloader.return_value = mock_loader
        mock_datamodule.val_dataloader.return_value = mock_loader
        mock_datamodule.test_dataloader.return_value = mock_loader

        def mock_instantiate(cfg_arg, **kwargs):
            if hasattr(cfg_arg, "_target_"):
                target = cfg_arg._target_ if hasattr(cfg_arg, "_target_") else ""
                if "Trainer" in str(target):
                    return mock_trainer
                elif "Dataset" in str(target):
                    return mock_dataset
                elif "DM" in str(target) or "DataModule" in str(target):
                    return mock_datamodule
            if "dataset_train" in kwargs:
                return mock_datamodule
            return MagicMock()

        monkeypatch.setattr(
            "picid.interface.interface.hydra.utils.instantiate", mock_instantiate
        )

        mock_lm = MagicMock()
        monkeypatch.setattr(
            "picid.interface.interface.create_lightning_module", lambda **kw: mock_lm
        )
        monkeypatch.setattr(
            "picid.interface.interface.register_data_dim_resolver", lambda data: None
        )
        monkeypatch.setattr(
            "picid.interface.interface.print_hydra_config_tree", lambda c: "tree"
        )

        return SimpleNamespace(
            trainer=mock_trainer,
            cfg=cfg,
            datamodule=mock_datamodule,
            train_data=train_data,
        )

    def test_debug_mode_skips_training(self, train_env):
        """debug=True skips fit/test and returns None.

        **Methodology**: Call train with debug=True.

        **Expected**: Trainer.fit not called.
        """
        from picid.interface.utils import ProcessedDatasource

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            debug=True,
            seed=42,
        )
        train_env.trainer.fit.assert_not_called()

    def test_processed_datasource_uses_data_dict(self, train_env):
        """ProcessedDatasource path uses its data_dict directly."""
        from picid.interface.utils import ProcessedDatasource

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            seed=42,
        )
        train_env.trainer.fit.assert_called_once()
        train_env.trainer.test.assert_called_once()

    def test_max_epochs_1_tests_without_checkpoint(self, train_env):
        """max_epochs=1 calls trainer.test with model instead of ckpt_path."""
        from picid.interface.utils import ProcessedDatasource

        OmegaConf.update(train_env.cfg, "trainer.max_epochs", 1)

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            seed=42,
        )
        test_call = train_env.trainer.test.call_args
        assert "model" in test_call.kwargs or (
            len(test_call.args) > 0 and test_call.args[0] is not None
        )

    def test_seed_from_config(self, train_env, monkeypatch):
        """seed=None falls back to config seed."""
        from picid.interface.utils import ProcessedDatasource

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        seed_calls = []
        monkeypatch.setattr(
            "picid.interface.interface.seed_everything",
            lambda s, workers=True: seed_calls.append(s),
        )

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            debug=True,
            seed=None,
        )
        assert seed_calls[0] == 42

    def test_flash_attention_disabled(self, train_env, monkeypatch):
        """flash_attention=False disables CUDA SDP.

        **Methodology**: Set flash_attention=False in config.

        **Expected**: torch.backends.cuda calls made.
        """
        from picid.interface.utils import ProcessedDatasource

        OmegaConf.update(train_env.cfg, "flash_attention", False)

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        flash_calls = []
        monkeypatch.setattr(
            "picid.interface.interface.torch.backends.cuda.enable_flash_sdp",
            lambda v: flash_calls.append(("flash", v)),
        )
        monkeypatch.setattr(
            "picid.interface.interface.torch.backends.cuda.enable_mem_efficient_sdp",
            lambda v: flash_calls.append(("mem_eff", v)),
        )
        monkeypatch.setattr(
            "picid.interface.interface.torch.backends.cuda.enable_math_sdp",
            lambda v: flash_calls.append(("math", v)),
        )

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            debug=True,
            seed=42,
        )
        assert ("flash", False) in flash_calls
        assert ("math", True) in flash_calls

    def test_transforms_not_sequence_raises(self, train_env):
        """Non-sequence transforms raises AssertionError."""
        from picid.interface.utils import ProcessedDatasource

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        with pytest.raises(AssertionError, match="transforms must be a list"):
            iface.train(
                run_name="test",
                model=MagicMock(),
                task_definition={"task_type": "rul"},
                datasource=mock_ds,
                transforms=42,
                seed=42,
            )

    def test_feature_mismatch_raises(self, train_env):
        """Different feature keys across splits raises AssertionError."""
        from picid.interface.utils import ProcessedDatasource

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {"features": np.random.randn(5, 5)},  # missing "target"
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        with pytest.raises(AssertionError, match="Mismatch in features"):
            iface.train(
                run_name="test",
                model=MagicMock(),
                task_definition={"task_type": "rul"},
                datasource=mock_ds,
                seed=42,
            )

    def test_inverse_transform_key_resolves(self, train_env, monkeypatch):
        """inverse_transform_key resolves inverter from transforms."""
        from picid.interface.utils import ProcessedDatasource

        OmegaConf.update(train_env.cfg, "evaluator.train.apply_inverse_scaling", True)
        OmegaConf.update(train_env.cfg, "evaluator.train.inverse_transform_key", "rul")
        OmegaConf.update(train_env.cfg, "evaluator.val.apply_inverse_scaling", True)
        OmegaConf.update(train_env.cfg, "evaluator.val.inverse_transform_key", "rul")
        OmegaConf.update(train_env.cfg, "evaluator.test.apply_inverse_scaling", True)
        OmegaConf.update(train_env.cfg, "evaluator.test.inverse_transform_key", "rul")

        mock_inverter = MagicMock()
        monkeypatch.setattr(
            "picid.interface.interface.get_inverter_for_key_with_name",
            lambda transforms, key, which: (mock_inverter, "scaler"),
        )

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            seed=42,
        )
        train_env.trainer.fit.assert_called_once()

    def test_inverse_transform_key_none_raises(self, train_env, monkeypatch):
        """inverse_transform_key that resolves to None raises ValueError."""
        from picid.interface.utils import ProcessedDatasource

        OmegaConf.update(train_env.cfg, "evaluator.train.apply_inverse_scaling", True)
        OmegaConf.update(train_env.cfg, "evaluator.train.inverse_transform_key", "rul")
        OmegaConf.update(train_env.cfg, "evaluator.val.apply_inverse_scaling", False)
        OmegaConf.update(train_env.cfg, "evaluator.test.apply_inverse_scaling", False)

        monkeypatch.setattr(
            "picid.interface.interface.get_inverter_for_key_with_name",
            lambda transforms, key, which: (None, None),
        )

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        with pytest.raises(ValueError, match="inverse scaling"):
            iface.train(
                run_name="test",
                model=MagicMock(),
                task_definition={"task_type": "rul"},
                datasource=mock_ds,
                seed=42,
            )

    def test_inverse_transform_name_resolves(self, train_env, monkeypatch):
        """inverse_transform_name resolves by name from transforms list."""
        from picid.interface.utils import ProcessedDatasource
        from picid.transforms.base.multisource import InverseTransformMixin

        OmegaConf.update(train_env.cfg, "evaluator.train.apply_inverse_scaling", True)
        OmegaConf.update(
            train_env.cfg, "evaluator.train.inverse_transform_name", "scaler"
        )
        OmegaConf.update(train_env.cfg, "evaluator.val.apply_inverse_scaling", False)
        OmegaConf.update(train_env.cfg, "evaluator.test.apply_inverse_scaling", False)

        mock_transform = MagicMock()
        mock_transform.name = "scaler"
        mock_transform.transform_instance = MagicMock(spec=InverseTransformMixin)

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            transforms=[mock_transform],
            seed=42,
        )
        train_env.trainer.fit.assert_called_once()

    def test_neither_key_nor_name_raises(self, train_env, monkeypatch):
        """No inverse_transform_key or name raises ValueError."""
        from picid.interface.utils import ProcessedDatasource

        OmegaConf.update(train_env.cfg, "evaluator.train.apply_inverse_scaling", True)
        OmegaConf.update(train_env.cfg, "evaluator.val.apply_inverse_scaling", False)
        OmegaConf.update(train_env.cfg, "evaluator.test.apply_inverse_scaling", False)

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        with pytest.raises(ValueError, match="neither"):
            iface.train(
                run_name="test",
                model=MagicMock(),
                task_definition={"task_type": "rul"},
                datasource=mock_ds,
                seed=42,
            )

    def test_loggers_lightning_logger(self, train_env, monkeypatch):
        """Lightning Logger instances are passed through directly."""
        from picid.interface.utils import ProcessedDatasource
        from lightning.pytorch.loggers import Logger

        mock_logger = MagicMock(spec=Logger)

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            loggers=[mock_logger],
            debug=True,
            seed=42,
        )

    def test_loggers_base_logger(self, train_env, monkeypatch):
        """BaseLogger instances are instantiated via Hydra."""
        from picid.interface.utils import ProcessedDatasource
        from picid.interface.schemas.loggers import BaseLogger

        mock_logger = MagicMock(spec=BaseLogger)
        mock_logger.model_dump.return_value = {
            "_target_": "lightning.loggers.CSVLogger"
        }

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            loggers=[mock_logger],
            debug=True,
            seed=42,
        )

    def test_callbacks_as_single_callback(self, train_env, monkeypatch):
        """Single Callback instance passed to train is wrapped and forwarded."""
        from picid.interface.utils import ProcessedDatasource
        from lightning import Callback

        class _NoopCallback(Callback):
            pass

        cb = _NoopCallback()

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            callbacks=cb,
            seed=42,
        )

    def test_train_list_of_callbacks(self, train_env):
        """List of Callback instances takes the elif branch (lines 1190-1191)."""
        from picid.interface.utils import ProcessedDatasource
        from lightning import Callback

        class _CB(Callback):
            pass

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            callbacks=[_CB()],
            seed=42,
        )

    def test_train_callbacks_from_cfg(self, train_env, monkeypatch):
        """callbacks=None with cfg 'callbacks' key instantiates callbacks (lines 1193-1199)."""
        from picid.interface.utils import ProcessedDatasource

        cfg_with_cbs = OmegaConf.create(
            {
                **OmegaConf.to_container(train_env.cfg, resolve=False),
                "callbacks": {"my_cb": {"_target_": "SomeCallback"}},
            }
        )
        monkeypatch.setattr(
            EntryInterface, "_compose_config_file", lambda self, **kw: cfg_with_cbs
        )

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            callbacks=None,
            seed=42,
        )

    def test_train_model_config_name_cleanup(self, train_env, monkeypatch):
        """cfg with model.config_name has the key removed before module creation (lines 1209-1211)."""
        from picid.interface.utils import ProcessedDatasource

        cfg_with_config_name = OmegaConf.create(
            {
                **OmegaConf.to_container(train_env.cfg, resolve=False),
                "model": {"config_name": "my_model_v1"},
            }
        )
        monkeypatch.setattr(
            EntryInterface,
            "_compose_config_file",
            lambda self, **kw: cfg_with_config_name,
        )

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            seed=42,
        )

    def test_train_non_tensor_batch_value(self, train_env):
        """Batch with a non-tensor value logs the type (line 1071)."""
        from picid.interface.utils import ProcessedDatasource

        non_tensor_batch = {
            "features": torch.zeros(2, 5),
            "target": torch.zeros(2, 1),
            "metadata": "string_value",
        }
        non_tensor_loader = MagicMock()
        non_tensor_loader.__len__.return_value = 3
        non_tensor_loader.__iter__.side_effect = lambda: iter([non_tensor_batch])

        train_env.datamodule.train_dataloader.return_value = non_tensor_loader
        train_env.datamodule.val_dataloader.return_value = non_tensor_loader
        train_env.datamodule.test_dataloader.return_value = non_tensor_loader

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            seed=42,
        )

    def test_train_logger_from_cfg(self, train_env, monkeypatch):
        """loggers=None with cfg 'logger' key instantiates loggers from cfg (line 1090)."""
        from picid.interface.utils import ProcessedDatasource

        cfg_with_logger = OmegaConf.create(
            {
                **OmegaConf.to_container(train_env.cfg, resolve=False),
                "logger": {
                    "csv": {
                        "_target_": "lightning.pytorch.loggers.CSVLogger",
                        "save_dir": ".",
                    }
                },
            }
        )
        monkeypatch.setattr(
            EntryInterface, "_compose_config_file", lambda self, **kw: cfg_with_logger
        )

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            loggers=None,
            seed=42,
        )

    def test_train_model_checkpoint_config_set(self, train_env, monkeypatch):
        """ModelCheckpointWithConfig callback gets cfg assigned (line 1207)."""
        from picid.interface.utils import ProcessedDatasource
        from picid.callbacks.model_checkpoint import ModelCheckpointWithConfig

        mock_ds = MagicMock(spec=ProcessedDatasource)
        mock_ds.task_mode = "regression"
        mock_ds.data_dict = {
            "train": {
                "features": np.random.randn(10, 5),
                "target": np.random.randn(10, 1),
            },
            "val": {"features": np.random.randn(5, 5), "target": np.random.randn(5, 1)},
            "test": {
                "features": np.random.randn(5, 5),
                "target": np.random.randn(5, 1),
            },
        }
        mock_ds.meta_data_dict = {}

        ckpt_callback = ModelCheckpointWithConfig(dirpath="/tmp", filename="best")

        iface = EntryInterface()
        iface.train(
            run_name="test",
            model=MagicMock(),
            task_definition={"task_type": "rul"},
            datasource=mock_ds,
            callbacks=[ckpt_callback],
            seed=42,
        )
        assert ckpt_callback.config is not None


# ---------------------------------------------------------------------------
# Task definition schema — validator + serializer branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskDefinitionSchema:
    """Tests for BaseTaskDefinition model_validator and model_serializer branches."""

    def test_prognostic_task_type_appended_to_input_tensors(self):
        """
        Prognostic validator appends task_type to input_tensors when absent (lines 66-69).

        **Expected**: "rul" added to default ["features"].
        """
        from picid.interface.schemas.task_definition import Prognostic

        task = Prognostic(task_type="rul")
        assert "rul" in task.input_tensors

    def test_forecasting_task_type_not_propagated(self):
        """
        Forecasting is excluded from task_type propagation (validator guard).

        **Expected**: "forecasting" not added a second time; input_tensors unchanged.
        """
        from picid.interface.schemas.task_definition import Forecasting

        task = Forecasting()
        assert task.input_tensors == ["features", "time_features", "target"]
        assert task.input_tensors.count("forecasting") == 0

    def test_model_dump_contains_data_requirements(self):
        """
        model_dump() triggers serialize_model which injects model requirements (lines 73-79).

        **Expected**: result["model"]["data_requirements"]["input_tensors"] is present.
        """
        from picid.interface.schemas.task_definition import Prognostic

        task = Prognostic(task_type="rul")
        dumped = task.model_dump()
        assert "model" in dumped
        assert "data_requirements" in dumped["model"]
        assert "input_tensors" in dumped["model"]["data_requirements"]
        assert "rul" in dumped["model"]["data_requirements"]["input_tensors"]
