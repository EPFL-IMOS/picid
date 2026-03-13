"""Comprehensive tests for Hydra instantiator utilities.

This module tests the functions that instantiate PyTorch Lightning
callbacks and loggers from Hydra configuration.

PHM Context:
-----------
PHM experiments use various callbacks (checkpointing, early stopping)
and loggers (wandb, tensorboard) configured via Hydra.

Test Coverage Strategy:
----------------------
1. **Callback Instantiation**: Creating callbacks from config
2. **Logger Instantiation**: Creating loggers from config
3. **Error Handling**: Invalid configs, missing targets
4. **Edge Cases**: Empty configs, nested configs
"""

import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from picid.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)


class TestInstantiateCallbacks:
    """Tests for instantiate_callbacks function."""

    def test_empty_config_returns_empty_list(self):
        """Test that empty config returns empty list.

        **PHM Logic**: No callbacks configured means empty list.

        **Methodology**: Pass empty DictConfig.

        **Expected**: Empty list returned.

        Validates: Requirement IC-1.1 - Empty config handling
        """
        cfg = OmegaConf.create({})

        result = instantiate_callbacks(cfg)

        assert result == []

    def test_non_dictconfig_raises_error(self):
        """Test that non-DictConfig raises TypeError.

        **PHM Logic**: Only DictConfig is accepted.

        **Methodology**: Pass regular dict.

        **Expected**: TypeError raised.

        Validates: Requirement IC-1.2 - Type validation
        """
        cfg = {"callback1": {"_target_": "some.class"}}

        with pytest.raises(TypeError):
            instantiate_callbacks(cfg)

    @patch("picid.utils.instantiators.hydra.utils.instantiate")
    def test_valid_callback_instantiated(self, mock_instantiate):
        """Test that valid callback config is instantiated.

        **PHM Logic**: _target_ configs should be instantiated.

        **Methodology**: Mock hydra.instantiate, pass valid config.

        **Expected**: hydra.instantiate called with config.

        Validates: Requirement IC-1.3 - Valid callback instantiation
        """
        mock_callback = MagicMock()
        mock_instantiate.return_value = mock_callback

        cfg = OmegaConf.create(
            {
                "early_stopping": {
                    "_target_": "pytorch_lightning.callbacks.EarlyStopping",
                    "monitor": "val_loss",
                    "patience": 5,
                }
            }
        )

        result = instantiate_callbacks(cfg)

        # Should call instantiate
        assert mock_instantiate.called
        # Should return list with callback
        assert len(result) == 1

    def test_config_without_target_skipped(self):
        """Test that configs without _target_ are skipped.

        **PHM Logic**: Non-callback configs should be ignored.

        **Methodology**: Pass config without _target_.

        **Expected**: Empty list returned.

        Validates: Requirement IC-1.4 - Non-target config handling
        """
        cfg = OmegaConf.create(
            {
                "settings": {
                    "param1": "value1"
                    # No _target_ key
                }
            }
        )

        result = instantiate_callbacks(cfg)

        assert result == []


class TestInstantiateLoggers:
    """Tests for instantiate_loggers function."""

    def test_empty_config_returns_empty_list(self):
        """Test that empty config returns empty list.

        **PHM Logic**: No loggers configured means empty list.

        **Methodology**: Pass empty DictConfig.

        **Expected**: Empty list returned.

        Validates: Requirement IL-1.1 - Empty config handling
        """
        cfg = OmegaConf.create({})

        result = instantiate_loggers(cfg)

        assert result == []

    def test_non_dictconfig_raises_error(self):
        """Test that non-DictConfig raises TypeError.

        **PHM Logic**: Only DictConfig is accepted.

        **Methodology**: Pass regular dict.

        **Expected**: TypeError raised.

        Validates: Requirement IL-1.2 - Type validation
        """
        cfg = {"logger1": {"_target_": "some.logger"}}

        with pytest.raises(TypeError):
            instantiate_loggers(cfg)

    @patch("picid.utils.instantiators.hydra.utils.instantiate")
    def test_valid_logger_instantiated(self, mock_instantiate):
        """Test that valid logger config is instantiated.

        **PHM Logic**: _target_ configs should be instantiated.

        **Methodology**: Mock hydra.instantiate, pass valid config.

        **Expected**: hydra.instantiate called with config.

        Validates: Requirement IL-1.3 - Valid logger instantiation
        """
        mock_logger = MagicMock()
        mock_instantiate.return_value = mock_logger

        cfg = OmegaConf.create(
            {
                "tensorboard": {
                    "_target_": "pytorch_lightning.loggers.TensorBoardLogger",
                    "save_dir": "logs/",
                }
            }
        )

        result = instantiate_loggers(cfg)

        # Should call instantiate
        assert mock_instantiate.called
        # Should return list with logger
        assert len(result) == 1

    def test_config_without_target_skipped(self):
        """Test that configs without _target_ are skipped.

        **PHM Logic**: Non-logger configs should be ignored.

        **Methodology**: Pass config without _target_.

        **Expected**: Empty list returned.

        Validates: Requirement IL-1.4 - Non-target config handling
        """
        cfg = OmegaConf.create(
            {
                "settings": {
                    "param1": "value1"
                    # No _target_ key
                }
            }
        )

        result = instantiate_loggers(cfg)

        assert result == []


class TestInstantiatorsEdgeCases:
    """Edge case tests for instantiators."""

    def test_nested_dictconfig_without_target(self):
        """Test nested DictConfig without _target_.

        **PHM Logic**: Deeply nested configs may not have targets.

        **Methodology**: Pass deeply nested config.

        **Expected**: Graceful handling.

        Validates: Requirement IE-1.1 - Nested config handling
        """
        cfg = OmegaConf.create({"level1": {"level2": {"level3": {"value": 42}}}})

        # Should not crash
        result_callbacks = instantiate_callbacks(cfg)
        result_loggers = instantiate_loggers(cfg)

        assert isinstance(result_callbacks, list)
        assert isinstance(result_loggers, list)

    @patch("picid.utils.instantiators.hydra.utils.instantiate")
    def test_multiple_callbacks(self, mock_instantiate):
        """Test instantiation of multiple callbacks.

        **PHM Logic**: Multiple callbacks commonly configured.

        **Methodology**: Pass config with multiple callbacks.

        **Expected**: All callbacks instantiated.

        Validates: Requirement IE-1.2 - Multiple callback handling
        """
        mock_instantiate.side_effect = [MagicMock(), MagicMock()]

        cfg = OmegaConf.create(
            {
                "early_stopping": {
                    "_target_": "pytorch_lightning.callbacks.EarlyStopping",
                    "monitor": "val_loss",
                },
                "model_checkpoint": {
                    "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
                    "dirpath": "checkpoints/",
                },
            }
        )

        result = instantiate_callbacks(cfg)

        assert len(result) == 2
