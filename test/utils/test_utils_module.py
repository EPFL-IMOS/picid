"""Tests for picid.utils.utils (main utils module)."""

import pytest
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock

from picid.utils.utils import (
    extras,
    task_wrapper,
    get_metric_value,
    _get_hydra_overrides,
    _save_git_info_to_run,
    _save_uv_lock_to_run,
    _write_reproduce_guide,
)


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

        cfg = OmegaConf.create(
            {
                "paths": {"output_dir": "/tmp"},
                "debug": {"raise_inner_exception": False},
            }
        )
        with patch("picid.utils.utils.find_spec", return_value=None):
            m, o = task(cfg)
        assert m == {"m": 1}
        assert o == {"o": 2}

    def test_task_wrapper_logs_output_dir_on_exception(self):
        """Test that task_wrapper logs output dir in finally block even when task raises."""

        @task_wrapper
        def task(cfg):
            raise ValueError("task failed")

        cfg = OmegaConf.create(
            {
                "paths": {"output_dir": "/tmp/out"},
                "debug": {"raise_inner_exception": True},
            }
        )
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

        cfg = OmegaConf.create(
            {
                "paths": {"output_dir": "/tmp"},
                "debug": {"raise_inner_exception": True},
            }
        )
        with patch("picid.utils.utils.find_spec", return_value=None):
            with pytest.raises(RuntimeError, match="inner"):
                task(cfg)

    def test_task_wrapper_closes_wandb_when_installed(self):
        """Test that task_wrapper calls wandb.finish when wandb is installed and run exists."""

        @task_wrapper
        def task(cfg):
            return {"m": 1}, {"o": 2}

        cfg = OmegaConf.create(
            {
                "paths": {"output_dir": "/tmp"},
                "debug": {"raise_inner_exception": False},
            }
        )
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()  # truthy so wandb.finish branch runs
        with patch("picid.utils.utils.find_spec", return_value=MagicMock()):
            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                m, o = task(cfg)
        mock_wandb.finish.assert_called_once_with(exit_code=0)


class TestSaveUvLockToRun:
    def test_copies_uv_lock_to_output_dir(self, tmp_path):
        """_save_uv_lock_to_run copies uv.lock to run output dir when present."""
        root = tmp_path / "project"
        root.mkdir()
        uv_lock = root / "uv.lock"
        uv_lock.write_text("lock content")
        out_dir = tmp_path / "run_output"
        cfg = OmegaConf.create(
            {
                "paths": {"root_dir": str(root), "output_dir": str(out_dir)},
            }
        )
        _save_uv_lock_to_run(cfg)
        assert (out_dir / "uv.lock").read_text() == "lock content"

    def test_skips_when_no_paths(self):
        cfg = OmegaConf.create({})
        _save_uv_lock_to_run(cfg)  # no-op, no raise

    def test_skips_when_uv_lock_missing(self, tmp_path):
        out_dir = tmp_path / "run"
        out_dir.mkdir()
        cfg = OmegaConf.create(
            {
                "paths": {"root_dir": str(tmp_path), "output_dir": str(out_dir)},
            }
        )
        _save_uv_lock_to_run(cfg)
        assert not (out_dir / "uv.lock").exists()


class TestWriteReproduceGuide:
    def test_writes_reproduce_md_to_output_dir(self, tmp_path):
        """_write_reproduce_guide creates REPRODUCE.md with env setup, run command, debug config."""
        root = tmp_path / "project"
        root.mkdir()
        out_dir = tmp_path / "run_output"
        cfg = OmegaConf.create(
            {
                "paths": {"root_dir": str(root), "output_dir": str(out_dir)},
            }
        )
        _write_reproduce_guide(cfg)
        dest = out_dir / "REPRODUCE.md"
        assert dest.exists()
        content = dest.read_text()
        assert "How to Reproduce This Experiment" in content
        assert str(out_dir) in content
        assert "Environment setup" in content
        assert "Run the model" in content
        assert "Option A" in content
        assert "Option B" in content
        assert "reproduce_from_run" in content
        assert "Debug configuration" in content
        assert "uv sync" in content
        assert "picid/run.py" in content
        assert "debugpy" in content or '"type": "debugpy"' in content

    def test_skips_when_no_paths(self):
        cfg = OmegaConf.create({})
        _write_reproduce_guide(cfg)  # no-op, no raise

    def test_skips_when_no_output_dir(self):
        cfg = OmegaConf.create({"paths": {"root_dir": "/tmp"}})
        _write_reproduce_guide(cfg)  # no-op, no raise

    def test_output_dir_outside_root_uses_absolute_path(self, tmp_path):
        """When out_dir is not under root, relative_to raises ValueError → fallback to str(out_dir)."""
        root = tmp_path / "project"
        root.mkdir()
        out_dir = tmp_path / "elsewhere" / "run"
        cfg = OmegaConf.create(
            {"paths": {"root_dir": str(root), "output_dir": str(out_dir)}}
        )
        _write_reproduce_guide(cfg)
        dest = out_dir / "REPRODUCE.md"
        assert dest.exists()
        assert str(out_dir) in dest.read_text()

    def test_experiment_key_extracted_from_overrides(self, tmp_path):
        """experiment= override is parsed and appears in REPRODUCE.md."""
        root = tmp_path / "project"
        root.mkdir()
        out_dir = tmp_path / "run"
        cfg = OmegaConf.create(
            {"paths": {"root_dir": str(root), "output_dir": str(out_dir)}}
        )
        with patch(
            "picid.utils.utils._get_hydra_overrides",
            return_value=["experiment=my_exp", "trainer.max_epochs=5"],
        ):
            _write_reproduce_guide(cfg)
        content = (out_dir / "REPRODUCE.md").read_text()
        assert "my_exp" in content

    def test_oserror_on_write_is_swallowed(self, tmp_path):
        """OSError when writing REPRODUCE.md is caught and does not propagate."""
        root = tmp_path / "project"
        root.mkdir()
        out_dir = tmp_path / "run"
        cfg = OmegaConf.create(
            {"paths": {"root_dir": str(root), "output_dir": str(out_dir)}}
        )
        with patch(
            "picid.utils.utils.Path.write_text", side_effect=OSError("disk full")
        ):
            _write_reproduce_guide(cfg)  # must not raise


class TestGetHydraOverrides:
    def test_returns_empty_list_when_hydra_not_initialized(self):
        result = _get_hydra_overrides()
        assert result == []

    def test_returns_overrides_when_hydra_initialized(self):
        mock_hc = MagicMock()
        mock_hc.initialized.return_value = True
        mock_hc.get.return_value.overrides.task = ["trainer.max_epochs=1"]
        with patch("hydra.core.hydra_config.HydraConfig", mock_hc):
            result = _get_hydra_overrides()
        assert result == ["trainer.max_epochs=1"]


class TestSaveGitInfoToRun:
    def test_saves_run_metadata_yaml(self, tmp_path):
        out_dir = tmp_path / "run"
        cfg = OmegaConf.create(
            {"paths": {"root_dir": str(tmp_path), "output_dir": str(out_dir)}}
        )
        _save_git_info_to_run(cfg)
        assert (out_dir / "run_metadata.yaml").exists()

    def test_skips_when_no_paths(self):
        _save_git_info_to_run(OmegaConf.create({}))

    def test_root_defaults_to_cwd_when_dot(self, tmp_path):
        """root_dir='.' triggers Path.cwd() fallback."""
        out_dir = tmp_path / "run"
        cfg = OmegaConf.create({"paths": {"root_dir": ".", "output_dir": str(out_dir)}})
        _save_git_info_to_run(cfg)
        assert (out_dir / "run_metadata.yaml").exists()

    def test_oserror_on_save_is_swallowed(self, tmp_path):
        out_dir = tmp_path / "run"
        cfg = OmegaConf.create(
            {"paths": {"root_dir": str(tmp_path), "output_dir": str(out_dir)}}
        )
        with patch(
            "picid.utils.utils.OmegaConf.save", side_effect=OSError("disk full")
        ):
            _save_git_info_to_run(cfg)  # must not raise


class TestSaveUvLockToRunOSError:
    def test_oserror_on_copy_is_swallowed(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        (root / "uv.lock").write_text("lock")
        out_dir = tmp_path / "run"
        cfg = OmegaConf.create(
            {"paths": {"root_dir": str(root), "output_dir": str(out_dir)}}
        )
        with patch("picid.utils.utils.shutil.copy2", side_effect=OSError("disk full")):
            _save_uv_lock_to_run(cfg)  # must not raise
