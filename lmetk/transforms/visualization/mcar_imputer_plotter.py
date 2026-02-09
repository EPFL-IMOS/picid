import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, Optional, Tuple

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.data.data_objects.utils import convert_to_numpy

# Set backend for non-interactive environments
plt.switch_backend("Agg")

logger = logging.getLogger(__name__)


class MCARImputationPlotter(DenseTransform, NoFitMultiSourceTransformMixin):
    """
    Visualizes MCAR imputation results with high-precision, publication-ready alignment.
    """

    def __init__(
        self,
        saving_path: str,
        dataset_name: str = "Dataset",
        imputed_key: str = "features",
        corrupted_key: str = "features_corrupted",
        original_key: Optional[str] = "features_original",
        n_channels_to_plot: int = 3,
        unit_key: str = "unit_id",
        legend_channel: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.saving_path = saving_path
        self.dataset_name = dataset_name
        self.imputed_key = imputed_key
        self.corrupted_key = corrupted_key
        self.original_key = original_key
        self.n_channels_to_plot = n_channels_to_plot
        self.unit_key = unit_key
        self.legend_channel = legend_channel

        # --- Compact Publication Settings (Larger Fonts) ---
        plt.rcParams.update(
            {
                "figure.dpi": 300,
                "font.family": "serif",
                "font.size": 14,  # Base font
                "axes.labelsize": 14,  # Axis labels
                "axes.titlesize": 14,  # Subplot titles
                "xtick.labelsize": 12,  # Ticks
                "ytick.labelsize": 12,  # Ticks
                "legend.fontsize": 12,  # Legend
                "figure.titlesize": 16,  # Main title
                "figure.titleweight": "bold",
                "axes.grid": True,
                "grid.alpha": 0.4,
                "grid.linestyle": ":",
                "xtick.direction": "in",
                "ytick.direction": "in",
            }
        )

    def _get_unit_id(self, metadata: Dict) -> str:
        if metadata and self.unit_key in metadata:
            val = metadata[self.unit_key]
            if hasattr(val, "item"):
                return str(val.item())
            return str(val)
        return "UnknownUnit"

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict
    ) -> NamedTransformInput:
        if self.imputed_key not in data or self.corrupted_key not in data:
            logger.warning("Required keys for plotting not found.")
            return data

        # Data retrieval
        imputed = convert_to_numpy(data[self.imputed_key], ensure_2d=False)
        corrupted = convert_to_numpy(data[self.corrupted_key], ensure_2d=False)

        original = None
        if self.original_key and self.original_key in data:
            original = convert_to_numpy(data[self.original_key], ensure_2d=False)

        def standardize_shape(arr):
            if arr.ndim == 2:
                return arr[np.newaxis, :, :]
            elif arr.ndim == 3:
                return arr
            raise ValueError(f"Invalid shape {arr.shape}")

        imputed = standardize_shape(imputed)
        corrupted = standardize_shape(corrupted)
        if original is not None:
            original = standardize_shape(original)

        # Plot loop
        B, T, C = imputed.shape
        unit_id_base = self._get_unit_id(metadata)
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
        T, C = imputed.shape
        channels_to_plot = min(C, self.n_channels_to_plot)

        # Create figure
        fig, axes = plt.subplots(
            nrows=channels_to_plot,
            ncols=1,
            figsize=(10, 2.2 * channels_to_plot),
            sharex=True,
            gridspec_kw={"hspace": 0.15},
        )

        if channels_to_plot == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        time_steps = np.arange(T)

        for c in range(channels_to_plot):
            ax = axes[c]

            # --- Determine Global Min/Max for Tight Y-Axis ---
            valid_data = [imputed[:, c]]
            if original is not None:
                valid_data.append(original[:, c])

            all_vals = np.concatenate(valid_data)
            y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)

            # Add a tiny 2% padding Y-axis
            y_range = y_max - y_min if y_max != y_min else 1.0
            margin_y = y_range * 0.02
            ax.set_ylim(y_min - margin_y, y_max + margin_y)

            # --- 1. Plot Ground Truth (Gray Background) ---
            if original is not None:
                ax.plot(
                    time_steps,
                    original[:, c],
                    color="gray",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.6,
                    zorder=1,
                )

            # --- 2. Plot "Reconstructed" Gap (Red Middle Layer) ---
            ax.plot(
                time_steps,
                imputed[:, c],
                color="#d62728",  # Red
                linestyle="-",
                linewidth=1.5,
                zorder=2,
            )

            # --- 3. Plot "Observed" Data (Green Top Layer) ---
            ax.plot(
                time_steps,
                corrupted[:, c],
                color="#2ca02c",  # Green
                linestyle="-",
                linewidth=1.5,
                zorder=3,
            )

            # --- 4. Styling & Alignment ---
            ax.set_ylabel(f"Sensor {c+1}", fontweight="bold", fontsize=14)
            ax.tick_params(axis="y", labelsize=12)
            ax.grid(True, which="both", alpha=0.5)

            # --- X-AXIS PADDING (Tight) ---
            x_range = time_steps[-1] - time_steps[0]
            x_margin = x_range * 0.01
            ax.set_xlim(time_steps[0] - x_margin, time_steps[-1] + x_margin)

            # Force x-ticks visible
            ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)

            # --- LEGEND: Configurable Channel ---
            if c == self.legend_channel:
                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        lw=1.5,
                        linestyle="--",
                        alpha=0.6,
                        label="Ground Truth",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="#2ca02c",
                        lw=2.0,
                        linestyle="-",
                        label="Observed Data",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="#d62728",
                        lw=2.0,
                        linestyle="-",
                        label="Imputed Gap",
                    ),
                ]

                ax.legend(
                    handles=legend_elements,
                    loc="best",
                    frameon=True,
                    framealpha=0.9,
                    fontsize=12,
                    edgecolor="gray",
                )

        # Fix bottom axis
        axes[-1].tick_params(
            axis="x", which="both", bottom=True, labelbottom=True, labelsize=12
        )
        axes[-1].set_xlabel("Time (samples)", fontweight="bold", fontsize=14)

        # --- Alignment Logic ---
        fig.align_ylabels(axes)

        # --- TITLE LOGIC (ATTACHED TO SUBPLOT) ---
        if "UnknownUnit" in unit_id:
            title_text = self.dataset_name
        else:
            title_text = f"{self.dataset_name}: Unit {unit_id}"

        # CRITICAL FIX: Attach title to axes[0] instead of figure
        # pad=15 puts it exactly 15 points above the top spine
        axes[0].set_title(title_text, fontsize=16, fontweight="bold", pad=15)

        # --- Layout & Save ---
        # No more 'rect' adjustments needed because tight_layout handles axis titles automatically
        plt.tight_layout()

        save_dir = os.path.join(self.saving_path, "paper_plots")
        os.makedirs(save_dir, exist_ok=True)

        safe_unit_id = unit_id.replace(" ", "_").replace(":", "").replace("/", "_")
        safe_dataset = self.dataset_name.replace(" ", "_")
        base_name = f"{safe_dataset}_{safe_unit_id}"
        extension = ".png"

        filename = f"{base_name}{extension}"
        filepath = os.path.join(save_dir, filename)

        counter = 1
        while os.path.exists(filepath):
            filename = f"{base_name}_{counter}{extension}"
            filepath = os.path.join(save_dir, filename)
            counter += 1

        try:
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
        finally:
            plt.close(fig)
