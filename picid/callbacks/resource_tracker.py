"""Lightning callback for tracking compute resources and timing statistics."""

import time
import torch
import lightning.pytorch as pl
import numpy as np
from fvcore.nn import FlopCountAnalysis

# Try importing NVML for GPU Load tracking
try:
    import pynvml

    pynvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    print("⚠️ [ResourceTracker] 'nvidia-ml-py' not found. GPU Load tracking disabled.")


class ResourceTracker(pl.Callback):
    """
    Track compute resources and timing statistics during a run.

    Parameters
    ----------
    skip_first_n : int, default=10
        Number of initial batches to ignore when computing timing statistics.
    """

    def __init__(self, skip_first_n=10):
        self.skip_first_n = skip_first_n
        self.flops_calculated = False

        # Temp storage
        self.timers = {}
        self.recent_latencies = []

        # Global Accumulators
        self.stats = {
            "train_batch_times": [],
            "train_epoch_times": [],
            "val_batch_times": [],
            "val_epoch_times": [],
            "inference_latencies": [],
            "gpu_loads": [],  # <--- NEW: Stores utilization % per step
        }

    # --- Helpers ---
    def _sync(self):
        """Synchronize CUDA streams if a GPU is available."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _get_gpu_memory(self):
        """Return peak GPU memory allocated in MB, or 0.0 when no GPU is present."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    def _get_gpu_load(self):
        """
        Return the current GPU compute utilization percentage.

        Returns
        -------
        float
            Current GPU utilization in percent.
        """
        if not HAS_NVML or not torch.cuda.is_available():
            return 0.0
        try:
            # Get handle for the current device
            idx = torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            # Returns object with .gpu and .memory (we want .gpu usage %)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            return 0.0

    def _extract_features(self, batch):
        """
        Extract the input tensor from a batch structure.

        Parameters
        ----------
        batch : Any
            Batch structure supplied by the dataloader.

        Returns
        -------
        Any
            Extracted input tensor or the original batch fallback.
        """
        if isinstance(batch, dict):
            if "features" in batch:
                return batch["features"]
            if "context" in batch and isinstance(batch["context"], dict):
                return batch["context"].get("x", list(batch.values())[0])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k != "rul":
                    return v
        elif isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def _calculate_flops(self, pl_module, batch):
        """Compute and log forward-pass FLOPs once per run."""
        if self.flops_calculated:
            return
        try:
            x = self._extract_features(batch)
            flops_count = 0.0

            if hasattr(pl_module, "get_flops"):
                flops_count = pl_module.get_flops(x)
            else:
                flops = FlopCountAnalysis(pl_module, x)
                flops.unsupported_ops_warnings(False)
                flops_count = flops.total()

            if flops_count > 0:
                pl_module.log("efficiency/gflops", flops_count / 1e9)
                if pl_module.global_rank == 0:
                    print(f"   [ResourceTracker] FLOPs: {flops_count/1e9:.3f} G")
        except Exception:
            pass
        finally:
            self.flops_calculated = True

    # =========================================================
    # TRAINING HOOKS
    # =========================================================

    def on_train_epoch_start(self, trainer, pl_module):
        """Start the epoch timer and reset per-epoch GPU load accumulator."""
        self._sync()
        self.timers["train_epoch"] = time.time()
        # Reset GPU load stats for the new epoch
        self.stats["gpu_loads"] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record batch start time and trigger one-shot FLOPs calculation."""
        if not self.flops_calculated:
            self._calculate_flops(pl_module, batch)
        self._sync()
        self.timers["train_batch"] = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Record batch latency, GPU load, and log smoothed throughput."""
        self._sync()
        latency = time.time() - self.timers["train_batch"]

        # Capture GPU Load instantly after the batch finishes
        current_load = self._get_gpu_load()
        self.stats["gpu_loads"].append(current_load)

        if batch_idx > self.skip_first_n:
            self.stats["train_batch_times"].append(latency)

            # Smoothed Throughput
            x = self._extract_features(batch)
            bs = x.shape[0] if hasattr(x, "shape") else 1

            self.recent_latencies.append(latency)
            if len(self.recent_latencies) > 50:
                self.recent_latencies.pop(0)
            avg_lat = sum(self.recent_latencies) / len(self.recent_latencies)

            # Log metrics
            pl_module.log_dict(
                {
                    "efficiency/train_throughput": bs / avg_lat if avg_lat > 0 else 0.0,
                    "efficiency/gpu_load_step": current_load,  # Real-time look
                },
                prog_bar=True,
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch duration and average GPU load for the completed epoch."""
        self._sync()
        duration = time.time() - self.timers["train_epoch"]
        self.stats["train_epoch_times"].append(duration)

        # Calculate Average GPU Load for the entire epoch
        avg_gpu_load = (
            np.mean(self.stats["gpu_loads"]) if self.stats["gpu_loads"] else 0.0
        )

        pl_module.log_dict(
            {
                "time/train_epoch_duration_sec": duration,
                "efficiency/gpu_load_epoch_avg": avg_gpu_load,
            }
        )

    # =========================================================
    # VALIDATION & INFERENCE (Unchanged)
    # =========================================================

    def on_validation_epoch_start(self, trainer, pl_module):
        """Start the validation epoch timer."""
        self._sync()
        self.timers["val_epoch"] = time.time()

    def on_validation_batch_start(self, *args):
        """Start the validation batch timer."""
        self._sync()
        self.timers["val_batch"] = time.time()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args
    ):
        """Record validation batch latency after skipping warm-up batches."""
        self._sync()
        latency = time.time() - self.timers["val_batch"]
        if batch_idx > self.skip_first_n:
            self.stats["val_batch_times"].append(latency)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation epoch duration."""
        self._sync()
        duration = time.time() - self.timers["val_epoch"]
        self.stats["val_epoch_times"].append(duration)
        pl_module.log("time/val_epoch_duration_sec", duration)

    # Unified Inference
    def _on_inf_start(self):
        """Start the inference step timer."""
        self._sync()
        self.timers["inf_step"] = time.time()

    def _on_inf_end(self):
        """Record inference step latency."""
        self._sync()
        latency = time.time() - self.timers["inf_step"]
        self.stats["inference_latencies"].append(latency)

    def on_test_batch_start(self, *args, **kwargs):
        """Start inference timer at the beginning of a test batch."""
        self._on_inf_start()

    def on_test_batch_end(self, *args, **kwargs):
        """Record latency at the end of a test batch."""
        self._on_inf_end()

    def on_predict_batch_start(self, *args, **kwargs):
        """Start inference timer at the beginning of a predict batch."""
        self._on_inf_start()

    def on_predict_batch_end(self, *args, **kwargs):
        """Record latency at the end of a predict batch."""
        self._on_inf_end()

    def _log_inference_summary(self, trainer):
        """Log mean/std latency and peak VRAM to the trainer logger."""
        lats = np.array(self.stats["inference_latencies"])
        if len(lats) > 0:
            metrics = {
                "efficiency/inference_latency_mean_ms": np.mean(lats) * 1000,
                "efficiency/inference_latency_std_ms": np.std(lats) * 1000,
                "efficiency/peak_vram_mb": self._get_gpu_memory(),
            }
            trainer.logger.log_metrics(metrics)
            self.stats["inference_latencies"] = []

    def on_test_end(self, trainer, pl_module):
        """Log inference summary at the end of the test phase."""
        self._log_inference_summary(trainer)

    def on_predict_end(self, trainer, pl_module):
        """Log inference summary at the end of the predict phase."""
        self._log_inference_summary(trainer)

    def on_fit_end(self, trainer, pl_module):
        """
        Log the final summary, including GPU load statistics.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            Active trainer instance.
        pl_module : lightning.pytorch.LightningModule
            Active Lightning module.
        """
        summary = {}
        if self.stats["train_batch_times"]:
            summary["AvgTime/train_batch_mean_ms"] = (
                np.mean(self.stats["train_batch_times"]) * 1000
            )
            summary["AvgTime/train_batch_std_ms"] = (
                np.std(self.stats["train_batch_times"]) * 1000
            )
        if self.stats["train_epoch_times"]:
            summary["AvgTime/train_epoch_mean_sec"] = np.mean(
                self.stats["train_epoch_times"]
            )

        # Add Final Average GPU Load
        all_loads = self.stats.get("gpu_loads", [])
        if all_loads:
            summary["Efficiency/gpu_load_avg_percent"] = np.mean(all_loads)

        trainer.logger.log_metrics(summary)

        if pl_module.global_rank == 0:
            print("\n" + "=" * 40)
            print("⏱️  [ResourceTracker] Final Summary")
            for k, v in summary.items():
                print(f"   • {k}: {v:.4f}")
            print("=" * 40 + "\n")
