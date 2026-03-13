"""Tests for picid.utils.logging_utils."""

from unittest.mock import patch, MagicMock

from picid.utils.logging_utils import get_hydra_override, log_hyperparameters


class TestGetHydraOverride:
    def test_returns_none_when_hydra_not_initialized(self):
        with patch("picid.utils.logging_utils.HydraConfig") as m:
            m.initialized.return_value = False
            assert get_hydra_override("experiment") is None

    def test_returns_value_when_override_present(self):
        with patch("picid.utils.logging_utils.HydraConfig") as m:
            m.initialized.return_value = True
            m.get.return_value.overrides.task = ["experiment=my_exp"]
            assert get_hydra_override("experiment") == "my_exp"

    def test_returns_none_when_key_not_in_overrides(self):
        with patch("picid.utils.logging_utils.HydraConfig") as m:
            m.initialized.return_value = True
            m.get.return_value.overrides.task = ["other=val"]
            assert get_hydra_override("experiment") is None


class TestLogHyperparameters:
    def test_skips_when_no_logger(self):
        trainer = MagicMock()
        trainer.logger = None
        obj = {
            "cfg": {"a": 1},
            "model/params/total": 0,
            "model/params/trainable": 0,
            "trainer": trainer,
        }
        from lightning_utilities.core.rank_zero import rank_zero_only

        with patch.object(rank_zero_only, "rank", 0):
            with patch(
                "picid.utils.logging_utils.OmegaConf.to_container",
                return_value={"a": 1},
            ):
                log_hyperparameters(obj)

    def test_logs_when_logger_present(self):
        mock_logger = MagicMock()
        trainer = MagicMock(loggers=[mock_logger], logger=mock_logger)
        obj = {
            "cfg": {"a": 1},
            "model/params/total": 10,
            "model/params/trainable": 5,
            "trainer": trainer,
        }
        from lightning_utilities.core.rank_zero import rank_zero_only

        with patch.object(rank_zero_only, "rank", 0):
            with patch(
                "picid.utils.logging_utils.get_hydra_override", return_value="exp1"
            ):
                with patch(
                    "picid.utils.logging_utils.OmegaConf.to_container",
                    return_value={"a": 1},
                ):
                    log_hyperparameters(obj)
        mock_logger.log_hyperparams.assert_called_once()
