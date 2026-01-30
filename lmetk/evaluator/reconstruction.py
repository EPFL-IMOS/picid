import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, override
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.evaluator.default import DefaultEvaluator

logger = logging.getLogger(__name__)


class ReconstructionEvaluator(DefaultEvaluator):
    """
    Refactored Reconstruction Evaluator.
    Maintains legacy plotting and path logic while integrating with the updated
    DefaultEvaluator architecture.
    """

    def __init__(
        self,
        metric_names: List[str],
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        plot_reconstructions: bool = False,
        paths: Optional[Any] = None,
        **kwargs,
    ):
        # FIX 1: Prevent double-passing collect_predictions to super()
        collect_arg = kwargs.pop("collect_predictions", False)
        collect_predictions = collect_arg or plot_reconstructions

        # Non-interactive backend for headless environments
        matplotlib.use("Agg")

        self.plot_reconstructions = plot_reconstructions
        self.plots_out_path = None

        super().__init__(
            metric_names=metric_names,
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            save_predictions=save_predictions,
            collect_predictions=collect_predictions,
            paths=paths,
            task_type="regression",  # Reconstruction follows regression naming policy
            **kwargs,
        )

    def _plot_reconstructions(
        self, preds: np.ndarray, targets: np.ndarray, mode: str, epoch: int, step: int
    ) -> Optional[plt.Figure]:
        """Legacy plotting logic maintained for visualization and test coverage."""
        if not self.remote_logger:
            if step == 0:
                logger.warning(
                    "Remote logger is not configured. Skipping reconstruction plots."
                )
            return None

        num_samples = len(preds)
        indices = np.linspace(0, num_samples - 1, num=min(10, num_samples), dtype=int)

        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes_flat = axes.flatten()

        # Ensure 3D for loop safety
        if targets.ndim == 2:
            targets = np.expand_dims(targets, axis=-1)
        if preds.ndim == 2:
            preds = np.expand_dims(preds, axis=-1)

        num_dims = targets.shape[2]
        colors = plt.get_cmap("tab10").colors

        for i, idx in enumerate(indices):
            ax = axes_flat[i]
            dims_to_plot = range(min(num_dims, 10))

            for dim in dims_to_plot:
                color = colors[dim % len(colors)]
                ax.plot(
                    targets[idx, :, dim],
                    label=f"Target (Dim {dim})",
                    linestyle="--",
                    color=color,
                )
                ax.plot(
                    preds[idx, :, dim],
                    label=f"Prediction (Dim {dim})",
                    color=color,
                    alpha=0.7,
                )

            ax.set_title(f"Sample {idx}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            if i == 0:
                ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Reconstruction (Split: {mode}, Epoch: {epoch})")
        plt.tight_layout()
        return fig

    @override
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """Orchestrates truthful plotting followed by base metric computation."""

        # 1. Path Resolution - Matches test expectation for plot_dir
        if self.plots_out_path is None and self.paths:
            p_dir = getattr(self.paths, "plot_dir", None)
            # Robust check to avoid MagicMock auto-generation issues in tests
            if isinstance(p_dir, (str, Path)):
                self.plots_out_path = Path(p_dir) / mode
                self.plots_out_path.mkdir(parents=True, exist_ok=True)

        # 2. Plot Generation Logic (Side Effect)
        if self.plot_reconstructions and hasattr(self, "preds") and self.preds:
            # Use Truthful Scaling: Sourced from base class lists
            is_dual = self.scaling_wrapper.apply_inverse
            source_preds = self.scaled_preds if is_dual else self.preds
            source_targets = self.scaled_targets if is_dual else self.targets

            if source_preds and len(source_preds) > 0:
                preds_concat = np.concatenate(source_preds, axis=0)
                targets_concat = np.concatenate(source_targets, axis=0)

                fig_recons = self._plot_reconstructions(
                    preds_concat, targets_concat, mode, epoch, step
                )

                if fig_recons:
                    self.log_plot(
                        fig_recons, "reconstructions_overview", mode, epoch, step
                    )
                    if self.plots_out_path:
                        fig_recons.savefig(
                            self.plots_out_path
                            / f"reconstructions_overview_{epoch}_{step}.png"
                        )
                    plt.close(fig_recons)

        # 3. Delegate to Base for Metrics and NetCDF Saving
        return super().compute(mode, epoch=epoch, step=step)
