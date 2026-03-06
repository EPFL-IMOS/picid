"""Tests for picid.utils.utils (main utils module)."""
import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from picid.utils.utils import extras, task_wrapper, get_metric_value


class TestExtras:
    def test_returns_early_when_no_extras(self):
        cfg = OmegaConf.create({})
        extras(cfg)

    def test_ignores_warnings_when_set(self):
        cfg = OmegaConf.create({"extras": {"ignore_warnings": True}})
        with patch("picid.utils.utils.warnings.filterwarnings") as m:
            extras(cfg)
        m.assert_called_once_with("ignore")

    def test_enforce_tags_when_set(self):
        cfg = OmegaConf.create({"extras": {"enforce_tags": True}})
        with patch("picid.utils.utils.rich_utils.enforce_tags") as m:
            extras(cfg)
        m.assert_called_once_with(cfg, save_to_file=True)

    def test_print_config_when_set(self):
        """Test extras calls print_config_tree when cfg.extras.print_config is True."""
        cfg = OmegaConf.create({"extras": {"print_config": True}})
        with patch("picid.utils.utils.rich_utils.print_config_tree") as m:
            extras(cfg)
        m.assert_called_once_with(cfg, resolve=True, save_to_file=True)


class TestGetMetricValue:
    def test_returns_none_when_metric_name_none(self):
        assert get_metric_value({}, None) is None

    def test_raises_when_metric_not_found(self):
        with pytest.raises(Exception, match="Metric value not found"):
            get_metric_value({"other": 1.0}, "missing")

    def test_returns_value_when_found(self):
        import torch as _torch
        val = get_metric_value({"m1": _torch.tensor(0.5)}, "m1")
        assert val == 0.5


class TestTaskWrapper:
    def test_passthrough_on_success(self):
        @task_wrapper
        def task(cfg):
            return {"m": 1}, {"o": 2}

        cfg = OmegaConf.create({
            "paths": {"output_dir": "/tmp"},
            "debug": {"raise_inner_exception": False},
        })
        with patch("picid.utils.utils.find_spec", return_value=None):
            m, o = task(cfg)
        assert m == {"m": 1}
        assert o == {"o": 2}

    def test_task_wrapper_logs_output_dir_on_exception(self):
        """Test that task_wrapper logs output dir in finally block even when task raises."""
        @task_wrapper
        def task(cfg):
            raise ValueError("task failed")

        cfg = OmegaConf.create({
            "paths": {"output_dir": "/tmp/out"},
            "debug": {"raise_inner_exception": True},
        })
        with patch("picid.utils.utils.find_spec", return_value=None):
            with patch("picid.utils.utils.log") as mock_log:
                with pytest.raises(ValueError, match="task failed"):
                    task(cfg)
        mock_log.info.assert_any_call("Output dir: /tmp/out")

    def test_task_wrapper_raises_when_raise_inner_exception_true(self):
        """Test raise_inner_exception=True branch: exception is re-raised."""
        @task_wrapper
        def task(cfg):
            raise RuntimeError("inner")

        cfg = OmegaConf.create({
            "paths": {"output_dir": "/tmp"},
            "debug": {"raise_inner_exception": True},
        })
        with patch("picid.utils.utils.find_spec", return_value=None):
            with pytest.raises(RuntimeError, match="inner"):
                task(cfg)

    def test_task_wrapper_closes_wandb_when_installed(self):
        """Test that task_wrapper calls wandb.finish when wandb is installed and run exists."""
        @task_wrapper
        def task(cfg):
            return {"m": 1}, {"o": 2}

        cfg = OmegaConf.create({
            "paths": {"output_dir": "/tmp"},
            "debug": {"raise_inner_exception": False},
        })
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()  # truthy so wandb.finish branch runs
        with patch("picid.utils.utils.find_spec", return_value=MagicMock()):
            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                m, o = task(cfg)
        mock_wandb.finish.assert_called_once_with(exit_code=0)
