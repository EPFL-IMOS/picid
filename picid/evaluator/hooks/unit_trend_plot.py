from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from picid.evaluator.hooks.base import BaseEvalHook


class UnitTrendPlotHook(BaseEvalHook):
    """
    Plot unit-wise prediction-versus-target trends.

    Parameters
    ----------
    log_every_n_epochs : int, default=10
        Logging cadence for validation epochs.
    enable_subsampling : bool, default=True
        Whether to downsample very long sequences before plotting.
    subsample_threshold : int, default=2000
        Sequence length threshold after which subsampling is enabled.
    subsample_factor : int, default=10
        Step size used when subsampling is active.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        enable_subsampling: bool = True,
        subsample_threshold: int = 2000,
        subsample_factor: int = 10,
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.enable_subsampling = enable_subsampling
        self.subsample_threshold = subsample_threshold
        self.subsample_factor = subsample_factor

    def on_compute_end(
        self,
        results: Dict[str, float],
        evaluator: Any,
        mode: str,
        epoch: int,
        step: int,
    ) -> None:
        if not evaluator.remote_logger:
            return

        # Valid condition: mode isn't train, and if val, check frequency
        if mode == "train" or (mode == "val" and epoch % self.log_every_n_epochs != 0):
            return

        data = evaluator.buffer.get_all()
        if "unit_ids" not in data:
            return

        u_ids = data["unit_ids"]
        unique_units = np.unique(u_ids, axis=0) if u_ids.ndim > 1 else np.unique(u_ids)

        for u in unique_units:
            mask = (u_ids == u).all(axis=1) if u_ids.ndim > 1 else (u_ids == u)
            idx = np.where(mask)[0]

            p, t = data["preds"][idx].flatten(), data["targets"][idx].flatten()

            if self.enable_subsampling and p.size > self.subsample_threshold:
                p, t = p[:: self.subsample_factor], t[:: self.subsample_factor]

            fig, ax = plt.subplots()
            ax.plot(t, "--", label="GT")
            ax.plot(p, label="Pred")
            ax.set_title(f"Unit {u} Trend")
            ax.legend()

            unit_key = "_".join(map(str, u)) if isinstance(u, (np.ndarray, list)) else u
            evaluator.log_plot(fig, f"unit_{unit_key}_trend", mode, epoch, step)
            plt.close(fig)
