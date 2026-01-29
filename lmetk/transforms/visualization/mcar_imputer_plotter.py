import logging
import os
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.data.data_objects.utils import convert_to_numpy

# Set backend for non-interactive environments
plt.switch_backend("Agg")

logger = logging.getLogger(__name__)


class MCARImputationPlotter(DenseTransform, NoFitMultiSourceTransformMixin):
    """
    Visualizes the results of MCAR corruption and subsequent imputation.

    This transform generates plots comparing the Original (Clean) signal, the
    Corrupted (Missing) signal, and the Imputed (Repaired) signal. It highlights
    the specific points that were dropped and recovered to allow for visual
    inspection of imputation quality.

    It assumes the input dictionary contains keys for:
    1. The fully processed (imputed) data (e.g., 'features').
    2. The corrupted data with NaNs (e.g., 'features_corrupted').
    3. The original clean data (e.g., 'features_original').

    If 'features_original' is missing, it plots only the corrupted vs imputed comparison.

    Attributes:
        saving_path (str): Root directory to save the plots.
        imputed_key (str): Key for the final imputed data (default: 'features').
        corrupted_key (str): Key for the data with NaNs (default: 'features_corrupted').
        original_key (Optional[str]): Key for the clean ground truth (default: 'features_original').
        n_channels_to_plot (int): Max number of channels to plot per sample (default: 3).
        unit_key (str): Metadata key to identify the sample unit (default: 'unit_id').
    """

    def __init__(
        self,
        saving_path: str,
        imputed_key: str = "features",
        corrupted_key: str = "features_corrupted",
        original_key: Optional[str] = "features_original",
        n_channels_to_plot: int = 3,
        unit_key: str = "unit_id",
        **kwargs,
    ):
        """
        Initializes the MCARImputationPlotter.

        Args:
            saving_path: Directory to save plots.
            imputed_key: Data key for the repaired signal.
            corrupted_key: Data key for the signal with NaNs.
            original_key: Data key for the ground truth signal (optional).
            n_channels_to_plot: Number of channels to visualize per sample.
            unit_key: Metadata key for file naming.
            **kwargs: Additional arguments for BaseTransform.
        """
        super().__init__(**kwargs)
        self.saving_path = saving_path
        self.imputed_key = imputed_key
        self.corrupted_key = corrupted_key
        self.original_key = original_key
        self.n_channels_to_plot = n_channels_to_plot
        self.unit_key = unit_key

        # Standard plot settings
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "font.size": 10,
                "axes.grid": True,
                "grid.alpha": 0.5,
            }
        )

    def _get_unit_id(self, metadata: Dict) -> str:
        """Extracts unit ID from metadata for filenames."""
        if metadata and self.unit_key in metadata:
            val = metadata[self.unit_key]
            # Handle potential array/tensor wrapping
            if hasattr(val, "item"):
                return str(val.item())
            return str(val)
        return "UnknownUnit"

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict
    ) -> NamedTransformInput:
        """
        Generates comparison plots for the provided data batch.

        This method does NOT modify the data. It saves plots to disk and passes
        the data through unchanged.

        Args:
            data: Dictionary containing input arrays.
            metadata: Metadata dictionary.

        Returns:
            NamedTransformInput: The input data, unmodified.
        """
        # Validate keys exist
        if self.imputed_key not in data:
            logger.warning(f"Key '{self.imputed_key}' not found. Skipping plot.")
            return data

        if self.corrupted_key not in data:
            logger.warning(f"Key '{self.corrupted_key}' not found. Skipping plot.")
            return data

        # Convert to numpy and ensure (Batch, Time, Channels) format
        # If input is (Time, Channels), add Batch dim -> (1, T, C)
        imputed = convert_to_numpy(data[self.imputed_key], ensure_2d=False)
        corrupted = convert_to_numpy(data[self.corrupted_key], ensure_2d=False)

        original = None
        if self.original_key and self.original_key in data:
            original = convert_to_numpy(data[self.original_key], ensure_2d=False)

        # Standardization helper
        def standardize_shape(arr):
            if arr.ndim == 2:
                return arr[np.newaxis, :, :]
            elif arr.ndim == 3:
                return arr
            else:
                raise ValueError(f"Data must be 2D or 3D, got {arr.ndim}D")

        imputed = standardize_shape(imputed)
        corrupted = standardize_shape(corrupted)
        if original is not None:
            original = standardize_shape(original)

        # Plotting Loop (per batch)
        B, T, C = imputed.shape
        unit_id_base = self._get_unit_id(metadata)

        # Limit number of batches to plot to avoid IO bottlenecks (e.g. max 5)
        batches_to_plot = min(B, 5)

        for b in range(batches_to_plot):
            unit_id = f"{unit_id_base}_batch{b}"
            self._plot_sample(
                imputed[b],
                corrupted[b],
                original[b] if original is not None else None,
                unit_id,
            )

        return data

    def _plot_sample(
        self,
        imputed: np.ndarray,
        corrupted: np.ndarray,
        original: Optional[np.ndarray],
        unit_id: str,
    ):
        """
        Creates and saves a plot for a single sample (Time, Channels).
        Handles filename collisions by appending an incrementing ID.
        """
        T, C = imputed.shape
        channels_to_plot = min(C, self.n_channels_to_plot)

        fig, axes = plt.subplots(
            nrows=channels_to_plot,
            ncols=1,
            figsize=(10, 3 * channels_to_plot),
            sharex=True,
            squeeze=False,
        )
        axes = axes.flatten()
        time_steps = np.arange(T)

        for c in range(channels_to_plot):
            ax = axes[c]

            # 1. Plot Original (if available) - Gray dashed line
            if original is not None:
                ax.plot(
                    time_steps,
                    original[:, c],
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.6,
                    label="Original (Ground Truth)",
                )

            # 2. Plot Imputed Signal - Blue line
            ax.plot(
                time_steps,
                imputed[:, c],
                color="#1f77b4",
                linewidth=1.0,
                alpha=0.8,
                label="Imputed Signal",
            )

            # 3. Highlight Observed Points - Green Dots
            mask_observed = ~np.isnan(corrupted[:, c])
            if np.any(mask_observed):
                ax.scatter(
                    time_steps[mask_observed],
                    imputed[mask_observed, c],
                    color="#2ca02c",
                    s=10,
                    label="Observed Data",
                    zorder=3,
                )

            # 4. Highlight Missing/Imputed Regions - Red 'x'
            mask_missing = np.isnan(corrupted[:, c])
            if np.any(mask_missing):
                ax.scatter(
                    time_steps[mask_missing],
                    imputed[mask_missing, c],
                    color="#d62728",
                    marker="x",
                    s=20,
                    label="Imputed Points",
                    zorder=4,
                )

            ax.set_title(f"Channel {c}")
            ax.set_ylabel("Value")
            if c == 0:
                ax.legend(loc="upper right", framealpha=0.9)

        axes[-1].set_xlabel("Time Step")
        fig.suptitle(f"Imputation Analysis: {unit_id}", y=1.02, fontsize=12)
        plt.tight_layout()

        # Save Logic with Collision Handling
        save_dir = os.path.join(self.saving_path, "imputation_plots")
        os.makedirs(save_dir, exist_ok=True)

        base_name = unit_id
        extension = ".png"
        filename = f"{base_name}{extension}"
        filepath = os.path.join(save_dir, filename)

        # Check if file exists and append incremental ID if it does
        counter = 1
        while os.path.exists(filepath):
            filename = f"{base_name}_{counter}{extension}"
            filepath = os.path.join(save_dir, filename)
            counter += 1

        try:
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.debug(f"Saved imputation plot to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
        finally:
            plt.close(fig)
