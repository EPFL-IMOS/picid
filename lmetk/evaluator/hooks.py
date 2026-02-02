from abc import ABC
from typing import Any, Dict, Optional
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


class BaseEvalHook(ABC):
    """Base interface for all Evaluator Hooks."""

    def on_update_end(self, batch: Dict[str, Any], evaluator: Any):
        pass

    def on_compute_end(
        self,
        results: Dict[str, float],
        evaluator: Any,
        mode: str,
        epoch: int,
        step: int,
    ):
        pass


class SavePredictionsHook(BaseEvalHook):
    """Handles NetCDF saving logic. Task-aware for dimension naming."""

    def on_compute_end(self, results, evaluator, mode, epoch, step):
        if not evaluator.save_predictions:
            return

        data = evaluator.buffer.get_all()
        if not data:
            return

        # Handle Task-Specific Xarray Formatting
        xr_data = self._format_xarray(data, evaluator)

        p_dir = getattr(evaluator.paths, "eval_details", None)
        if p_dir:
            out_path = Path(p_dir) / mode
            out_path.mkdir(parents=True, exist_ok=True)
            xr.Dataset(xr_data).to_netcdf(out_path / "predictions.nc")

    def _format_xarray(
        self, data: Dict[str, np.ndarray], evaluator: Any
    ) -> Dict[str, Any]:
        """Maps raw buffer arrays to task-specific Xarray dimensions."""
        p = data["preds"]
        # Determine dimension names based on task
        if evaluator.task_type == "classification" and p.ndim > 2:
            d_names = ["sample", "time", "class_logits"]
            t_names = ["sample", "time", "class_label"]
        else:
            d_names = ["sample"] + [f"dim{i}" for i in range(1, p.ndim)]
            t_names = d_names

        res = {"preds": (d_names, p), "targets": (t_names, data["targets"])}

        if "norm_preds" in data:
            res["preds_normalized"] = (d_names, data["norm_preds"])
            res["targets_normalized"] = (t_names, data["norm_targets"])

        if "unit_ids" in data:
            u = data["unit_ids"]
            u_dims = ["sample"] + [f"unit_dim{i}" for i in range(1, u.ndim)]
            res["unit_ids"] = (u_dims, u)

        return res


class ReconstructionPlotHook(BaseEvalHook):
    """Hook for 5x2 Reconstruction comparison plots."""

    def on_compute_end(self, results, evaluator, mode, epoch, step):
        if (
            not getattr(evaluator, "plot_reconstructions", False)
            or not evaluator.remote_logger
        ):
            return

        data = evaluator.buffer.get_all()
        if not data.get("preds") is not None:
            return

        preds, targets = data["preds"], data["targets"]
        num_samples = len(preds)
        indices = np.linspace(0, num_samples - 1, num=min(10, num_samples), dtype=int)

        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes_flat = axes.flatten()

        for i, idx in enumerate(indices):
            ax = axes_flat[i]
            # Plot first dimension
            ax.plot(targets[idx, :, 0], "--", label="Target")
            ax.plot(preds[idx, :, 0], label="Pred")
            ax.set_title(f"Idx {idx}")

        plt.tight_layout()
        evaluator.log_plot(fig, "reconstructions", mode, epoch, step)
        plt.close(fig)


class UnitTrendPlotHook(BaseEvalHook):
    """Hook for per-unit trend plots."""

    def on_compute_end(self, results, evaluator, mode, epoch, step):
        if (
            not getattr(evaluator, "log_fit_vs_target_curves", False)
            or not evaluator.remote_logger
        ):
            return

        # Frequency Logic
        if (
            mode == "val"
            and epoch % getattr(evaluator, "log_image_every_n_epochs", 10) != 0
        ):
            return

        data = evaluator.buffer.get_all()
        if "unit_ids" not in data:
            return

        u_ids = data["unit_ids"]
        unique_units = np.unique(u_ids, axis=0) if u_ids.ndim > 1 else np.unique(u_ids)

        for u in unique_units:
            idx = (
                np.where((u_ids == u).all(axis=1))[0]
                if u_ids.ndim > 1
                else np.where(u_ids == u)[0]
            )
            fig, ax = plt.subplots()
            ax.plot(data["targets"][idx].flatten(), "--", label="True")
            ax.plot(data["preds"][idx].flatten(), label="Pred")
            ax.set_title(f"Unit {u}")
            evaluator.log_plot(fig, f"unit_{u}_trend", mode, epoch, step)
            plt.close(fig)
