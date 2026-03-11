"""Tests for picid.utils.pylogger."""

import pytest
from unittest.mock import patch

from picid.utils.pylogger import RankedLogger


class TestRankedLogger:
    def test_repr(self):
        logger = RankedLogger("test.module", rank_zero_only=False)
        r = repr(logger)
        assert "RankedLogger" in r
        assert "test.module" in r
        assert "rank_zero_only" in r

    def test_log_raises_when_rank_not_set(self):
        logger = RankedLogger("test", rank_zero_only=False)
        with patch.object(logger, "isEnabledFor", return_value=True):
            with patch("picid.utils.pylogger.rank_zero_only") as m:
                m.rank = None
                with pytest.raises(RuntimeError, match="rank_zero_only.rank"):
                    logger.log(20, "msg")

    def test_log_calls_underlying_when_rank_zero_and_rank_zero_only(self):
        logger = RankedLogger("test", rank_zero_only=True)
        with patch.object(logger, "isEnabledFor", return_value=True):
            with patch("picid.utils.pylogger.rank_zero_only") as m:
                m.rank = 0
                with patch(
                    "picid.utils.pylogger.rank_prefixed_message",
                    side_effect=lambda msg, rank: msg,
                ):
                    with patch.object(logger.logger, "log") as mock_log:
                        logger.log(20, "msg")
                        mock_log.assert_called_once()
