"""
Tests for picid.callbacks.resource_tracker (ResourceTracker).

Validates FLOPs/throughput/GPU/timing hooks and helpers. Aligns with PHM
pipeline efficiency logging (docs: runai.md, pipeline training).
"""

import torch
from unittest.mock import MagicMock, patch

from picid.callbacks.resource_tracker import ResourceTracker


# -----------------------------------------------------------------------------
# Test: ResourceTracker init and helpers
# -----------------------------------------------------------------------------


class TestResourceTrackerInit:
    """Validates init state and skip_first_n."""

    def test_init_default_skip_first_n(self):
        """Doc: resource_tracker.py - skip_first_n defaults to 10."""
        cb = ResourceTracker()
        assert cb.skip_first_n == 10
        assert cb.flops_calculated is False
        assert "train_batch_times" in cb.stats
        assert "gpu_loads" in cb.stats

    def test_init_custom_skip_first_n(self):
        """Doc: resource_tracker.py - skip_first_n is configurable."""
        cb = ResourceTracker(skip_first_n=5)
        assert cb.skip_first_n == 5


class TestResourceTrackerHelpers:
    """Validates _sync, _get_gpu_memory, _get_gpu_load, _extract_features."""

    def test_sync_no_op_when_cuda_not_available(self):
        """Doc: resource_tracker.py - _sync only synchronizes when cuda available."""
        cb = ResourceTracker()
        cb._sync()

    def test_get_gpu_memory_returns_zero_without_cuda(self):
        """Doc: resource_tracker.py - _get_gpu_memory returns 0.0 when cuda not available."""
        cb = ResourceTracker()
        with patch("torch.cuda.is_available", return_value=False):
            assert cb._get_gpu_memory() == 0.0

    def test_get_gpu_load_returns_zero_without_nvml_or_cuda(self):
        """Doc: resource_tracker.py - _get_gpu_load returns 0.0 when no NVML or cuda."""
        cb = ResourceTracker()
        # HAS_NVML may be True/False depending on env; without cuda we expect 0
        with patch("torch.cuda.is_available", return_value=False):
            assert cb._get_gpu_load() == 0.0

    def test_extract_features_dict_with_features(self, phm_batch_features):
        """Doc: resource_tracker.py - if 'features' in batch, return batch['features']."""
        cb = ResourceTracker()
        out = cb._extract_features(phm_batch_features)
        assert out is phm_batch_features["features"]

    def test_extract_features_dict_with_context_x(self, phm_batch_context_x):
        """Doc: resource_tracker.py - if context dict with 'x', return context['x']."""
        cb = ResourceTracker()
        out = cb._extract_features(phm_batch_context_x)
        assert out is phm_batch_context_x["context"]["x"]

    def test_extract_features_dict_first_tensor_not_rul(self, phm_batch_other_tensor):
        """Doc: resource_tracker.py - else first tensor with k != 'rul'."""
        cb = ResourceTracker()
        out = cb._extract_features(phm_batch_other_tensor)
        assert out is phm_batch_other_tensor["other"]

    def test_extract_features_list_returns_first(self, phm_batch_list):
        """Doc: resource_tracker.py - elif list/tuple return batch[0]."""
        cb = ResourceTracker()
        out = cb._extract_features(phm_batch_list)
        assert out is phm_batch_list[0]

    def test_extract_features_fallback_returns_batch(self):
        """Doc: resource_tracker.py - else return batch."""
        cb = ResourceTracker()
        batch = "other"
        assert cb._extract_features(batch) == "other"


# -----------------------------------------------------------------------------
# Test: _calculate_flops (branch coverage)
# -----------------------------------------------------------------------------


class TestResourceTrackerFlops:
    """Validates _calculate_flops early return and get_flops path."""

    def test_calculate_flops_skips_after_first_call(
        self, mock_pl_module, phm_batch_features
    ):
        """Doc: resource_tracker.py - if flops_calculated, return immediately."""
        cb = ResourceTracker()
        cb.flops_calculated = True
        cb._calculate_flops(mock_pl_module, phm_batch_features)
        mock_pl_module.log.assert_not_called()

    def test_calculate_flops_uses_get_flops_when_present(
        self, mock_pl_module, phm_batch_features
    ):
        """Doc: resource_tracker.py - if pl_module has get_flops, use it."""
        mock_pl_module.get_flops = MagicMock(return_value=1e9)
        cb = ResourceTracker()
        cb._calculate_flops(mock_pl_module, phm_batch_features)
        mock_pl_module.log.assert_called_once()
        # First positional arg is the metric key
        logged_key = mock_pl_module.log.call_args[0][0]
        assert logged_key == "efficiency/gflops"

    def test_calculate_flops_sets_flops_calculated_on_exception(self, mock_pl_module):
        """Doc: resource_tracker.py - finally block sets flops_calculated=True."""
        cb = ResourceTracker()
        batch = {}  # no tensor -> FlopCountAnalysis may fail
        cb._calculate_flops(mock_pl_module, batch)
        assert cb.flops_calculated is True

    def test_calculate_flops_uses_flop_count_analysis_when_no_get_flops(
        self, phm_batch_features
    ):
        """Doc: resource_tracker.py - else use FlopCountAnalysis(pl_module, x)."""
        pl_module = torch.nn.Linear(5, 1)  # real module without get_flops
        pl_module.log = MagicMock()
        pl_module.global_rank = 0
        cb = ResourceTracker()
        x = phm_batch_features["features"]
        batch = {"features": x}
        cb._calculate_flops(pl_module, batch)
        pl_module.log.assert_called_once()
        assert cb.flops_calculated


# -----------------------------------------------------------------------------
# Test: Training and validation hooks (state changes)
# -----------------------------------------------------------------------------


class TestResourceTrackerHooks:
    """Validates epoch/batch hooks update timers and stats."""

    def test_on_train_epoch_start_resets_gpu_loads(
        self, mock_trainer_global_zero, mock_pl_module
    ):
        """Doc: resource_tracker.py - on_train_epoch_start resets gpu_loads."""
        cb = ResourceTracker()
        cb.stats["gpu_loads"] = [1, 2, 3]
        cb.on_train_epoch_start(mock_trainer_global_zero, mock_pl_module)
        assert cb.stats["gpu_loads"] == []
        assert "train_epoch" in cb.timers

    def test_on_train_batch_end_skips_first_n(
        self, mock_trainer_global_zero, mock_pl_module, phm_batch_features
    ):
        """Doc: resource_tracker.py - batch times appended only when batch_idx > skip_first_n."""
        cb = ResourceTracker(skip_first_n=2)
        cb.on_train_batch_start(
            mock_trainer_global_zero, mock_pl_module, phm_batch_features, 0
        )
        cb.on_train_batch_end(
            mock_trainer_global_zero, mock_pl_module, None, phm_batch_features, 0
        )
        assert len(cb.stats["train_batch_times"]) == 0
        cb.on_train_batch_end(
            mock_trainer_global_zero, mock_pl_module, None, phm_batch_features, 5
        )
        assert len(cb.stats["train_batch_times"]) == 1

    def test_on_validation_epoch_end_logs_duration(
        self, mock_trainer_global_zero, mock_pl_module
    ):
        """Doc: resource_tracker.py - on_validation_epoch_end logs val_epoch_duration."""
        cb = ResourceTracker()
        cb.on_validation_epoch_start(mock_trainer_global_zero, mock_pl_module)
        cb.on_validation_epoch_end(mock_trainer_global_zero, mock_pl_module)
        assert len(cb.stats["val_epoch_times"]) == 1
        mock_pl_module.log.assert_called()
        calls = [str(c) for c in mock_pl_module.log.call_args_list]
        assert any("val_epoch_duration" in c for c in calls)

    def test_on_test_batch_start_end_appends_inference_latency(
        self, mock_trainer_global_zero, mock_pl_module, phm_batch_features
    ):
        """Doc: resource_tracker.py - test batch hooks append to inference_latencies."""
        cb = ResourceTracker()
        cb.on_test_batch_start(
            mock_trainer_global_zero, mock_pl_module, phm_batch_features, 0
        )
        cb.on_test_batch_end(
            mock_trainer_global_zero, mock_pl_module, None, phm_batch_features, 0
        )
        assert len(cb.stats["inference_latencies"]) == 1

    def test_on_fit_end_logs_summary(self, mock_trainer_global_zero, mock_pl_module):
        """Doc: resource_tracker.py - on_fit_end logs AvgTime and GPU load summary."""
        cb = ResourceTracker()
        cb.stats["train_batch_times"] = [0.1, 0.2]
        cb.stats["train_epoch_times"] = [1.0]
        cb.stats["gpu_loads"] = [50, 60]
        cb.on_fit_end(mock_trainer_global_zero, mock_pl_module)
        mock_trainer_global_zero.logger.log_metrics.assert_called_once()
        kwargs = mock_trainer_global_zero.logger.log_metrics.call_args[0][0]
        assert (
            "AvgTime/train_batch_mean_ms" in kwargs
            or "AvgTime/train_epoch_mean_sec" in kwargs
            or "Efficiency/gpu_load_avg" in kwargs
        )

    def test_on_test_end_logs_inference_summary(
        self, mock_trainer_global_zero, mock_pl_module
    ):
        """Doc: resource_tracker.py - on_test_end logs inference latency mean/std."""
        cb = ResourceTracker()
        cb.stats["inference_latencies"] = [0.01, 0.02]
        cb.on_test_end(mock_trainer_global_zero, mock_pl_module)
        mock_trainer_global_zero.logger.log_metrics.assert_called_once()

    def test_on_predict_end_logs_inference_summary(
        self, mock_trainer_global_zero, mock_pl_module
    ):
        """Doc: resource_tracker.py - on_predict_end same as on_test_end."""
        cb = ResourceTracker()
        cb.stats["inference_latencies"] = [0.01]
        cb.on_predict_end(mock_trainer_global_zero, mock_pl_module)
        mock_trainer_global_zero.logger.log_metrics.assert_called_once()


# -----------------------------------------------------------------------------
# Test: CUDA paths and NVML (mocked)
# -----------------------------------------------------------------------------


class TestCudaPaths:
    """Validates GPU paths that require CUDA to be available (mocked)."""

    def test_sync_calls_synchronize_when_cuda_available(self):
        """_sync() calls torch.cuda.synchronize when cuda is available (line 49)."""
        cb = ResourceTracker()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.synchronize") as mock_sync,
        ):
            cb._sync()
        mock_sync.assert_called_once()

    def test_get_gpu_memory_returns_mb_when_cuda_available(self):
        """_get_gpu_memory() returns allocated MB when cuda is available (line 53)."""
        cb = ResourceTracker()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.max_memory_allocated", return_value=1024 * 1024 * 512),
        ):
            result = cb._get_gpu_memory()
        assert result == 512.0


class TestNvmlPath:
    """Validates _get_gpu_load paths when pynvml is mocked available."""

    def test_get_gpu_load_returns_utilization_via_nvml(self):
        """HAS_NVML=True + cuda + nvml mock → returns util.gpu (lines 69-73)."""
        import picid.callbacks.resource_tracker as rt_mod

        cb = ResourceTracker()
        mock_util = MagicMock()
        mock_util.gpu = 75

        with (
            patch.object(rt_mod, "HAS_NVML", True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch.object(rt_mod, "pynvml") as mock_pynvml,
        ):
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
            mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
            result = cb._get_gpu_load()

        assert result == 75

    def test_get_gpu_load_returns_zero_on_nvml_exception(self):
        """NVML raises → _get_gpu_load returns 0.0 (line 75)."""
        import picid.callbacks.resource_tracker as rt_mod

        cb = ResourceTracker()
        with (
            patch.object(rt_mod, "HAS_NVML", True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch.object(rt_mod, "pynvml") as mock_pynvml,
        ):
            mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError(
                "nvml err"
            )
            result = cb._get_gpu_load()

        assert result == 0.0


class TestSlidingWindow:
    """Validates the recent_latencies sliding window."""

    def test_recent_latencies_capped_at_50(
        self, mock_trainer_global_zero, mock_pl_module, phm_batch_features
    ):
        """Calling on_train_batch_end() > 50 times keeps recent_latencies ≤ 50 (line 159)."""
        cb = ResourceTracker(skip_first_n=0)
        for i in range(55):
            cb.on_train_batch_start(
                mock_trainer_global_zero, mock_pl_module, phm_batch_features, i
            )
            cb.on_train_batch_end(
                mock_trainer_global_zero, mock_pl_module, None, phm_batch_features, i
            )
        assert len(cb.recent_latencies) <= 50


class TestEpochHooks:
    """Validates on_train_epoch_end gpu_load averaging branches."""

    def test_epoch_end_with_gpu_loads_logged(
        self, mock_trainer_global_zero, mock_pl_module
    ):
        """on_train_epoch_end with recorded gpu_loads → log_dict includes avg (lines 172-181)."""
        cb = ResourceTracker()
        cb.timers["train_epoch"] = __import__("time").time()
        cb.stats["gpu_loads"] = [40.0, 60.0]
        cb.on_train_epoch_end(mock_trainer_global_zero, mock_pl_module)
        mock_pl_module.log_dict.assert_called()
        call_kwargs = mock_pl_module.log_dict.call_args[0][0]
        assert "efficiency/gpu_load_epoch_avg" in call_kwargs
        assert abs(call_kwargs["efficiency/gpu_load_epoch_avg"] - 50.0) < 1e-6

    def test_epoch_end_with_empty_gpu_loads_uses_zero(
        self, mock_trainer_global_zero, mock_pl_module
    ):
        """on_train_epoch_end with empty gpu_loads → avg_gpu_load = 0.0."""
        cb = ResourceTracker()
        cb.timers["train_epoch"] = __import__("time").time()
        cb.stats["gpu_loads"] = []
        cb.on_train_epoch_end(mock_trainer_global_zero, mock_pl_module)
        call_kwargs = mock_pl_module.log_dict.call_args[0][0]
        assert call_kwargs["efficiency/gpu_load_epoch_avg"] == 0.0


class TestValidationBatchHooks:
    """Validates validation batch timing hooks."""

    def test_validation_batch_start_sets_timer(self):
        """on_validation_batch_start sets timers['val_batch'] (lines 197-198)."""
        cb = ResourceTracker()
        cb.on_validation_batch_start()
        assert "val_batch" in cb.timers

    def test_validation_batch_end_appends_latency(
        self, mock_trainer_global_zero, mock_pl_module, phm_batch_features
    ):
        """on_validation_batch_end with batch_idx > skip_first_n appends latency (lines 203-206)."""
        cb = ResourceTracker(skip_first_n=2)
        cb.on_validation_batch_start()
        cb.on_validation_batch_end(
            mock_trainer_global_zero, mock_pl_module, None, phm_batch_features, 5
        )
        assert len(cb.stats["val_batch_times"]) == 1

    def test_validation_batch_end_skips_early_batches(
        self, mock_trainer_global_zero, mock_pl_module, phm_batch_features
    ):
        """on_validation_batch_end with batch_idx ≤ skip_first_n does not append."""
        cb = ResourceTracker(skip_first_n=5)
        cb.on_validation_batch_start()
        cb.on_validation_batch_end(
            mock_trainer_global_zero, mock_pl_module, None, phm_batch_features, 3
        )
        assert len(cb.stats["val_batch_times"]) == 0


class TestPredictBatchHooks:
    """Validates on_predict_batch_start/end delegate to inference timing (lines 231, 234)."""

    def test_predict_batch_start_sets_inf_timer(self):
        """on_predict_batch_start sets timers['inf_step'] via _on_inf_start (line 231)."""
        cb = ResourceTracker()
        cb.on_predict_batch_start()
        assert "inf_step" in cb.timers

    def test_predict_batch_end_appends_inference_latency(self):
        """on_predict_batch_end appends to inference_latencies via _on_inf_end (line 234)."""
        cb = ResourceTracker()
        cb.on_predict_batch_start()
        cb.on_predict_batch_end()
        assert len(cb.stats["inference_latencies"]) == 1


class TestSummaryEdgeCases:
    """Validates on_fit_end and _log_inference_summary edge cases."""

    def test_fit_end_with_empty_stats_does_not_raise(
        self, mock_trainer_global_zero, mock_pl_module
    ):
        """on_fit_end with empty stats → log_metrics called with empty dict (lines 265-280)."""
        cb = ResourceTracker()
        cb.on_fit_end(mock_trainer_global_zero, mock_pl_module)
        mock_trainer_global_zero.logger.log_metrics.assert_called_once_with({})

    def test_log_inference_summary_skips_when_empty(self, mock_trainer_global_zero):
        """_log_inference_summary with no latencies → log_metrics NOT called (line 231)."""
        cb = ResourceTracker()
        cb.stats["inference_latencies"] = []
        cb._log_inference_summary(mock_trainer_global_zero)
        mock_trainer_global_zero.logger.log_metrics.assert_not_called()
