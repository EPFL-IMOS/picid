from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from picid.evaluator.hooks.base import BaseEvalHook


class ReconstructionPlotHook(BaseEvalHook):
    """Plot reconstruction versus target signals for inspection."""

    def on_compute_end(
        self,
        results: Dict[str, float],
        evaluator: Any,
        mode: str,
        epoch: int,
        step: int,
    ) -> None:
        if (
            not getattr(evaluator, "plot_reconstructions", False)
            or not evaluator.remote_logger
        ):
            return

        data = evaluator.buffer.get_all()
        if not data or data.get("preds") is None:
            return

        preds, targets = data["preds"], data["targets"]
        indices = np.linspace(0, len(preds) - 1, num=min(10, len(preds)), dtype=int)

        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes_flat = axes.flatten()

        for i, idx in enumerate(indices):
            ax = axes_flat[i]
            # Plotting first channel/dim for comparison
            ax.plot(targets[idx, :, 0], "--", color="gray", label="Target")
            ax.plot(preds[idx, :, 0], color="blue", alpha=0.7, label="Reconstruction")
            ax.set_title(f"Sample {idx}")

        plt.tight_layout()
        evaluator.log_plot(fig, "reconstructions_overview", mode, epoch, step)
        plt.close(fig)
