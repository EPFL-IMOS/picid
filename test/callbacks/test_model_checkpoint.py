"""
Tests for picid.callbacks.model_checkpoint (ModelCheckpointWithConfig).

Validates that the callback saves the model checkpoint and, on the main process,
writes config to a JSON file once. Aligns with PHM pipeline checkpointing and
docs (config persistence for reproducibility).
"""

import os
import json
import pytest

from picid.callbacks.model_checkpoint import ModelCheckpointWithConfig


# -----------------------------------------------------------------------------
# Test: ModelCheckpointWithConfig init and config contract
# -----------------------------------------------------------------------------


class TestModelCheckpointWithConfigInit:
    """Validates init and that config must be set before _save_checkpoint."""

    def test_init_sets_config_filename_default(self):
        """Doc: model_checkpoint.py - config_filename defaults to 'hparams.json'."""
        cb = ModelCheckpointWithConfig()
        assert cb.config_filename == "hparams.json"
        assert cb.config is None

    def test_init_sets_custom_config_filename(self):
        """Doc: model_checkpoint.py - config_filename is configurable."""
        cb = ModelCheckpointWithConfig(config_filename="experiment_config.json")
        assert cb.config_filename == "experiment_config.json"

    def test_save_checkpoint_asserts_config_set(
        self, temp_checkpoint_dir, mock_trainer_global_zero
    ):
        """Doc: model_checkpoint.py - assert self.config is not None before writing."""
        cb = ModelCheckpointWithConfig(dirpath=temp_checkpoint_dir)
        assert cb.config is None
        filepath = os.path.join(temp_checkpoint_dir, "epoch=0.ckpt")
        with pytest.raises(AssertionError, match="Config is not set"):
            cb._save_checkpoint(mock_trainer_global_zero, filepath)


# -----------------------------------------------------------------------------
# Test: _save_checkpoint writes config JSON on main process
# -----------------------------------------------------------------------------


class TestModelCheckpointWithConfigSave:
    """Validates config file is written once and verbose message."""

    def test_save_checkpoint_writes_config_when_file_not_exists(
        self, temp_checkpoint_dir, mock_trainer_global_zero
    ):
        """
        **PHM Logic**: Reproducibility requires saving hyperparameters/config
        alongside the checkpoint. Doc: model_checkpoint.py - on global_zero,
        if config path does not exist, write config JSON.
        """
        cb = ModelCheckpointWithConfig(dirpath=temp_checkpoint_dir)
        cb.config = {"lr": 0.01, "batch_size": 32}
        filepath = os.path.join(temp_checkpoint_dir, "epoch=0.ckpt")
        # Create a minimal checkpoint file so parent _save_checkpoint does not fail
        with open(filepath, "w") as f:
            f.write("dummy")
        cb._save_checkpoint(mock_trainer_global_zero, filepath)
        config_path = os.path.join(temp_checkpoint_dir, "hparams.json")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            data = json.load(f)
        assert data["lr"] == 0.01
        assert data["batch_size"] == 32

    def test_save_checkpoint_skips_write_when_config_file_exists(
        self, temp_checkpoint_dir, mock_trainer_global_zero
    ):
        """Doc: model_checkpoint.py - if config path exists, do not overwrite."""
        cb = ModelCheckpointWithConfig(dirpath=temp_checkpoint_dir)
        cb.config = {"lr": 0.02}
        config_path = os.path.join(temp_checkpoint_dir, "hparams.json")
        with open(config_path, "w") as f:
            json.dump({"existing": True}, f)
        filepath = os.path.join(temp_checkpoint_dir, "epoch=0.ckpt")
        with open(filepath, "w") as f:
            f.write("dummy")
        cb._save_checkpoint(mock_trainer_global_zero, filepath)
        with open(config_path) as f:
            data = json.load(f)
        assert data.get("existing") is True
        assert "lr" not in data

    def test_save_checkpoint_verbose_prints_message(
        self, temp_checkpoint_dir, mock_trainer_global_zero
    ):
        """Doc: model_checkpoint.py - if self.verbose, trainer.print(config_path)."""
        cb = ModelCheckpointWithConfig(dirpath=temp_checkpoint_dir, verbose=True)
        cb.config = {"a": 1}
        filepath = os.path.join(temp_checkpoint_dir, "epoch=0.ckpt")
        with open(filepath, "w") as f:
            f.write("dummy")
        cb._save_checkpoint(mock_trainer_global_zero, filepath)
        mock_trainer_global_zero.print.assert_called()
        call_args = mock_trainer_global_zero.print.call_args[0][0]
        assert "hparams.json" in call_args or "Configuration saved" in call_args

    def test_save_checkpoint_not_global_zero_does_not_write_config(
        self, temp_checkpoint_dir, mock_trainer_not_global_zero
    ):
        """Doc: model_checkpoint.py - only global_rank==0 writes config."""
        cb = ModelCheckpointWithConfig(dirpath=temp_checkpoint_dir)
        cb.config = {"lr": 0.01}
        filepath = os.path.join(temp_checkpoint_dir, "epoch=0.ckpt")
        with open(filepath, "w") as f:
            f.write("dummy")
        cb._save_checkpoint(mock_trainer_not_global_zero, filepath)
        config_path = os.path.join(temp_checkpoint_dir, "hparams.json")
        assert not os.path.exists(config_path)
