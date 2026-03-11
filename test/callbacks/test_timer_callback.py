"""
Tests for picid.callbacks.timer_callback (PipelineTimer).

Validates timing hooks and _log_averages. Aligns with PHM pipeline
benchmarking (docs: runai.md, pipeline training/validation/test stages).
"""

import numpy as np
from unittest.mock import MagicMock

from picid.callbacks.timer_callback import PipelineTimer


# -----------------------------------------------------------------------------
# Test: PipelineTimer init
# -----------------------------------------------------------------------------


class TestPipelineTimerInit:
    """Validates init state."""

    def test_init_sums_and_counts(self):
        """Doc: timer_callback.py - sums and counts for each stage."""
        cb = PipelineTimer()
        assert "train_batch" in cb.sums
        assert "val_epoch" in cb.sums
        assert "test_batch" in cb.sums
        assert cb.counts["train_batch"] == 0
        assert cb.skip_first_n == 10

    def test_init_times_empty(self):
        """Doc: timer_callback.py - times dict for active timers."""
        cb = PipelineTimer()
        assert cb.times == {}


# -----------------------------------------------------------------------------
# Test: _start_timer, _end_timer
# -----------------------------------------------------------------------------


class TestPipelineTimerHelpers:
    """Validates _start_timer and _end_timer."""

    def test_start_timer_sets_time(self):
        """Doc: timer_callback.py - _start_timer(stage) sets self.times[stage]."""
        cb = PipelineTimer()
        cb._start_timer("train_batch")
        assert "train_batch" in cb.times
        assert isinstance(cb.times["train_batch"], (int, float))

    def test_end_timer_accumulates_when_stage_started(self):
        """Doc: timer_callback.py - _end_timer appends elapsed to sums and increments count."""
        cb = PipelineTimer()
        cb._start_timer("train_batch")
        cb._end_timer("train_batch")
        assert len(cb.sums["train_batch"]) == 1
        assert cb.counts["train_batch"] == 1

    def test_end_timer_no_op_when_stage_not_started(self):
        """Doc: timer_callback.py - if stage not in self.times, _end_timer does not append."""
        cb = PipelineTimer()
        cb._end_timer("train_batch")
        assert len(cb.sums["train_batch"]) == 0
        assert cb.counts["train_batch"] == 0


# -----------------------------------------------------------------------------
# Test: _log_averages
# -----------------------------------------------------------------------------


class TestPipelineTimerLogAverages:
    """Validates _log_averages branches: test vs non-test, counts>0, logger None."""

    def test_log_averages_non_test_uses_skip_first_n(self):
        """Doc: timer_callback.py - for non-test stages, mean/std use sums[skip_first_n:]."""
        cb = PipelineTimer()
        cb.skip_first_n = 2
        cb.sums["train_batch"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        cb.counts["train_batch"] = 5
        trainer = MagicMock()
        trainer.logger = MagicMock()
        cb._log_averages(trainer)
        trainer.logger.log_metrics.assert_called_once()
        metrics = trainer.logger.log_metrics.call_args[0][0]
        assert "AvgTime/train_batch_mean" in metrics
        assert "AvgTime/train_batch_std" in metrics
        # Mean should be over [0.3, 0.4, 0.5]
        np.testing.assert_almost_equal(
            metrics["AvgTime/train_batch_mean"], np.mean([0.3, 0.4, 0.5])
        )

    def test_log_averages_test_stage_no_skip(self):
        """Doc: timer_callback.py - for test stages, mean over all sums, std=0."""
        cb = PipelineTimer()
        cb.sums["test_batch"] = [0.1, 0.2]
        cb.counts["test_batch"] = 2
        trainer = MagicMock()
        trainer.logger = MagicMock()
        cb._log_averages(trainer)
        metrics = trainer.logger.log_metrics.call_args[0][0]
        assert "AvgTime/test_batch_mean" in metrics
        assert metrics["AvgTime/test_batch_std"] == 0.0

    def test_log_averages_skips_zero_count_stages(self):
        """Doc: timer_callback.py - only stages with counts[stage] > 0 are logged."""
        cb = PipelineTimer()
        cb.sums["train_batch"] = []
        cb.counts["train_batch"] = 0
        trainer = MagicMock()
        trainer.logger = MagicMock()
        cb._log_averages(trainer)
        metrics = trainer.logger.log_metrics.call_args[0][0]
        assert "AvgTime/train_batch_mean" not in metrics

    def test_log_averages_no_op_when_logger_none(self):
        """Doc: timer_callback.py - if trainer.logger is None, do not call log_metrics."""
        cb = PipelineTimer()
        cb.sums["train_batch"] = [0.1]
        cb.counts["train_batch"] = 1
        trainer = MagicMock()
        trainer.logger = None
        cb._log_averages(trainer)
        # No logger to call
        assert not hasattr(trainer.logger, "log_metrics") or trainer.logger is None


# -----------------------------------------------------------------------------
# Test: All stage hooks call _start_timer / _end_timer
# -----------------------------------------------------------------------------


class TestPipelineTimerHooks:
    """Validates each hook updates sums/counts."""

    def test_train_batch_hooks(self):
        """Doc: timer_callback.py - on_train_batch_start/end use train_batch."""
        cb = PipelineTimer()
        cb.on_train_batch_start(None, None)
        cb.on_train_batch_end(None, None)
        assert len(cb.sums["train_batch"]) == 1
        assert cb.counts["train_batch"] == 1

    def test_train_epoch_hooks(self):
        """Doc: timer_callback.py - on_train_epoch_start/end use train_epoch."""
        cb = PipelineTimer()
        cb.on_train_epoch_start(None, None)
        cb.on_train_epoch_end(None, None)
        assert len(cb.sums["train_epoch"]) == 1

    def test_val_batch_and_epoch_hooks(self):
        """Doc: timer_callback.py - val_batch and val_epoch hooks."""
        cb = PipelineTimer()
        cb.on_validation_batch_start(None, None)
        cb.on_validation_batch_end(None, None)
        cb.on_validation_epoch_start(None, None)
        cb.on_validation_epoch_end(None, None)
        assert len(cb.sums["val_batch"]) == 1
        assert len(cb.sums["val_epoch"]) == 1

    def test_test_hooks(self):
        """Doc: timer_callback.py - test_batch and test_epoch hooks."""
        cb = PipelineTimer()
        cb.on_test_batch_start(None, None)
        cb.on_test_batch_end(None, None)
        cb.on_test_epoch_start(None, None)
        cb.on_test_epoch_end(None, None)
        assert len(cb.sums["test_batch"]) == 1
        assert len(cb.sums["test_epoch"]) == 1

    def test_on_train_end_calls_log_averages(self):
        """Doc: timer_callback.py - on_train_end calls _log_averages(trainer)."""
        cb = PipelineTimer()
        cb.sums["train_batch"] = [0.1]
        cb.counts["train_batch"] = 1
        trainer = MagicMock()
        trainer.logger = MagicMock()
        cb.on_train_end(trainer, None)
        trainer.logger.log_metrics.assert_called_once()
