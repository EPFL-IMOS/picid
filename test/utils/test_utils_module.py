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
