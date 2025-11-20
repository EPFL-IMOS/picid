from typing import List, Optional, Dict, Any, override
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import logging

from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.evaluator.default import DefaultEvaluator

logger = logging.getLogger(__name__)


class ReconstructionEvaluator(DefaultEvaluator):
    """
    Evaluator specifically designed for reconstruction tasks (e.g., Autoencoders).

    It extends the DefaultEvaluator to include functionality for visualizing
    reconstruction performance by plotting predictions against targets. It handles
    both normalized and denormalized plotting based on the scaling configuration.
    """

    def __init__(
        self,
        metric_names: List[str],
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        plot_reconstructions: bool = False,
        paths: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the ReconstructionEvaluator.

        Parameters
        ----------
        metric_names : List[str]
            List of metric names to compute (e.g., 'mse', 'mae').
        inverse_transform : Optional[InverseTransformMixin]
            The scaler object used to inverse transform predictions/targets.
        apply_inverse_scaling : bool
            Whether to apply inverse scaling before computing metrics and plotting.
            If True, plots will show real-world values; otherwise, normalized values.
        save_predictions : bool
            Whether to save the raw predictions to disk (netCDF).
        plot_reconstructions : bool
            Whether to generate and log plots comparing input vs. reconstruction.
            If True, this forces `collect_predictions` to be True.
        paths : Optional[str]
            Configuration object containing path information (specifically `plot_dir`).
        **kwargs : dict
            Additional arguments passed to DefaultEvaluator.
        """
        # FIX 1: Handle 'collect_predictions' argument safely.
        # We must pop it from kwargs because we pass it explicitly to super().__init__.
        # If we leave it in kwargs, super() receives it twice (once named, once in **kwargs).
        collect_arg = kwargs.pop("collect_predictions", False)

        # Force data collection if plotting is required, otherwise respect user config.
        collect_predictions = collect_arg or plot_reconstructions

        # Use non-interactive backend to prevent display errors on headless servers/clusters.
        matplotlib.use("Agg")

        self.plot_reconstructions = plot_reconstructions
        self.plots_out_path = None

        super().__init__(
            metric_names=metric_names,
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            save_predictions=save_predictions,
            collect_predictions=collect_predictions,  # Passed explicitly here
            paths=paths,
            **kwargs,  # Safe to pass remaining kwargs
        )

    def _plot_reconstructions(
        self, preds: np.ndarray, targets: np.ndarray, mode: str, epoch: int, step: int
    ) -> Optional[plt.Figure]:
        """
        Generates a matplotlib Figure comparing targets and predictions.

        It selects a subsample (up to 10) of the batch and plots the signal dimensions.
        Handles both Univariate (2D) and Multivariate (3D) data shapes safely.

        Parameters
        ----------
        preds : np.ndarray
            Predictions array, shape (N, T, D) or (N, T).
        targets : np.ndarray
            Targets array, shape (N, T, D) or (N, T).
        mode : str
            Current split mode (e.g., 'val', 'test').
        epoch : int
            Current epoch number.
        step : int
            Current step number.

        Returns
        -------
        Optional[plt.Figure]
            The generated figure, or None if the remote logger is missing.
        """
        if not self.remote_logger:
            # Log warning only on the first step (step 0) to avoid log spamming
            if step == 0:
                logger.warning(
                    "Remote logger is not configured. Skipping reconstruction plots."
                )
            return None

        # Select up to 10 evenly spaced indices for plotting.
        # If num_samples < 10, it selects all available samples.
        num_samples = len(preds)
        indices = np.linspace(0, num_samples - 1, num=min(10, num_samples), dtype=int)

        # Create a 5x2 grid. Note: If batch < 10, some subplots might be empty or overwritten,
        # but the loop logic handles specific indices.
        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes = axes.flatten()

        # FIX 2: Dimension Safety for Univariate Data.
        # If data is (N, T), accessing .shape[2] later would cause an IndexError.
        # We expand it to (N, T, 1) to standardize the loop logic.
        if targets.ndim == 2:
            targets = np.expand_dims(targets, axis=-1)
        if preds.ndim == 2:
            preds = np.expand_dims(preds, axis=-1)

        num_dims = targets.shape[2]
        colors = plt.get_cmap("tab10").colors

        for i, idx in enumerate(indices):
            # Since indices length is guaranteed <= 10, and axes length is 10,
            # simple indexing is safe here.
            ax = axes[i]

            # Limit dimensions to plot to avoid cluttered graphs (max 10 dims)
            dims_to_plot = range(min(num_dims, 10))

            for dim in dims_to_plot:
                # Cycle colors if there are more dimensions than colors in the palette
                color = colors[dim % len(colors)]

                # Plot Target (Dashed Line)
                ax.plot(
                    targets[idx, :, dim],
                    label=f"Target (Dim {dim})",
                    linestyle="--",
                    color=color,
                )
                # Plot Prediction (Solid Line)
                ax.plot(
                    preds[idx, :, dim],
                    label=f"Prediction (Dim {dim})",
                    color=color,
                    alpha=0.7,
                )

            ax.set_title(f"Sample {idx}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")

            # Only show legend on the first plot to save visual space
            if i == 0:
                ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Reconstruction (Split: {mode}, Epoch: {epoch})")
        plt.tight_layout()
        return fig

    @override
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """
        Orchestrates plot generation, saving, and metric computation.

        Overrides DefaultEvaluator.compute to inject the plotting logic before
        the standard metric calculation and file saving.
        """

        # 1. Setup output path for local plot saving if not already set
        if self.plots_out_path is None:
            if self.paths and getattr(self.paths, "plot_dir", None):
                self.plots_out_path = Path(self.paths.plot_dir) / mode
                self.plots_out_path.mkdir(parents=True, exist_ok=True)

        # 2. Generate Plots (Side Effect)
        if self.plot_reconstructions:

            # FIX 3: Select correct data scale (Normalized vs Denormalized).
            # If inverse scaling is ON, users want to see Real-world values (scaled_preds).
            # If OFF, they see the raw model output (preds).
            use_denormalized = self.scaling_wrapper.apply_inverse

            source_preds = self.scaled_preds if use_denormalized else self.preds
            source_targets = self.scaled_targets if use_denormalized else self.targets

            # Ensure we actually have data collected before trying to concatenate
            if source_preds and len(source_preds) > 0:
                preds = np.concatenate(source_preds, axis=0)
                targets = np.concatenate(source_targets, axis=0)

                fig_recons = self._plot_reconstructions(
                    preds, targets, mode, epoch, step
                )

                if fig_recons:
                    # Log to remote tracker (e.g. WandB/MLFlow) via Base method
                    self.log_plot(
                        fig_recons, "reconstructions_overview", mode, epoch, step
                    )

                    # Save Locally if path is valid
                    if self.plots_out_path:
                        fig_recons.savefig(
                            self.plots_out_path
                            / f"reconstructions_overview_{epoch}_{step}.png"
                        )

                    # Close figure to free memory
                    plt.close(fig_recons)

        # 3. Delegate to Base for standard Metric Computation & NetCDF Saving
        return super().compute(mode, epoch=epoch, step=step)
