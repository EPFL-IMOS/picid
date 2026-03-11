import logging
from typing import Dict, Optional, Tuple, override
import numpy as np
import awkward as ak
import os
import math
import matplotlib.pyplot as plt

from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base.base_transform import RaggedOrDenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin


# Import your awkward utils
from picid.utils.awkward_utils import (
    ak_flatten_variable_dims,
    get_ak_shape,
)

# Set backend for non-interactive environments
plt.switch_backend("Agg")

logger = logging.getLogger(__name__)


class FeatureTargetPlotter(NoFitPerSegmentMixin, RaggedOrDenseTransform):
    """
    A pass-through transform that plots features and/or targets
    from `awkward` arrays.

    This transform is for debugging and analysis. It intelligently
    handles ragged arrays and broadcasting between features and targets
    before flattening them into 2D numpy arrays for plotting.

    It handles multiple data shapes, including:
    - (N, var, F) features and (N, 1) targets (broadcasts target).
    - (N, var, F) features and (N, var, 1) targets (flattens both).
    - (N, F) features and (N, 1) targets (standard numpy).

    Plots are saved to structured subdirectories:
    `saving_path/features_and_targets/`
    `saving_path/features_only/`
    `saving_path/targets_only/`
    `saving_path/original_targets/` (if broadcasting occurs)
    """

    def __init__(
        self,
        saving_path: str,
        unit_key: str,
        features_key: Optional[str] = "features",
        target_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the AwkwardFeatureTargetPlotter.

        Parameters
        ----------
        saving_path : str
            The root directory path to save images.
        unit_key : str
            The key in the `metadata` dict to find the
            unit identifier (e.g., 'unit_id').
        features_key : str, optional
            The key in the `data` dict for the features.
            If None, only targets will be plotted.
        target_key : str, optional
            The key in the `data` dict for the targets.
            If None, only features will be plotted.
        **kwargs
            Additional arguments for BaseTransform.
        """
        super().__init__(**kwargs)
        self.saving_path = saving_path
        self.unit_key = unit_key
        self.features_key = features_key
        self.target_key = target_key

        # Determine the plotting regime
        if self.features_key and self.target_key:
            self.regime = "features_and_targets"
        elif self.features_key:
            self.regime = "features_only"
        elif self.target_key:
            self.regime = "targets_only"
        else:
            raise ValueError(
                "At least one of 'features_key' or 'target_key' must be provided."
            )

        # Set plot rc settings
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "font.size": 10,
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Palatino", "serif"],
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.grid": True,
                "grid.linewidth": 0.5,
                "grid.alpha": 0.6,
            }
        )
        logger.debug(
            f"AwkwardFeatureTargetPlotter initialized in '{self.regime}' mode."
        )

    def _get_unit_id(self, metadata: Dict) -> str:
        """Helper to safely extract the unit ID from metadata for logging/filenames."""
        if metadata and self.unit_key in metadata:
            unit_id = metadata[self.unit_key]
            if (
                isinstance(unit_id, (tuple, list, np.ndarray))
                and unit_id.ndim > 0
                and len(unit_id) > 1
            ):
                return "_".join(map(str, unit_id))
            else:
                return str(unit_id).replace(" ", "_")
        return "UnknownUnit"

    def _prepare_data(
        self, data: NamedTransformInput
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Handles shape checking, broadcasting, and flattening of awkward arrays.
        Returns (feat_flat, targ_flat, orig_targ_np).
        """
        feat_data, targ_data = None, None
        feat_shape, targ_shape = [], []
        orig_targ_np = None  # <-- New variable to store original target

        # --- 1. Get Data (as ak.Array) ---
        if self.features_key:
            if self.features_key not in data:
                raise KeyError(f"features_key '{self.features_key}' not in data.")
            feat_data = ak.Array(data[self.features_key])
            feat_shape = get_ak_shape(feat_data)

        if self.target_key:
            if self.target_key not in data:
                raise KeyError(f"target_key '{self.target_key}' not in data.")
            targ_data = ak.Array(data[self.target_key])
            targ_shape = get_ak_shape(targ_data)

        # --- 2. Handle Scenarios ---
        is_scenario_1 = (
            feat_data is not None
            and len(feat_shape) == 3
            and feat_shape[1] == "var"
            and targ_data is not None
            and len(targ_shape) < 3  # (N,) or (N, 1)
            and feat_shape[0] == targ_shape[0]  # Match N dimension
        )

        feat_flat, targ_flat = None, None

        if is_scenario_1:
            logger.debug("Running broadcast scenario: (N, var, F) + (N, 1)")

            feat_flat = ak.to_numpy(ak_flatten_variable_dims(feat_data, axis=1))

            counts_np = ak.to_numpy(ak.num(feat_data, axis=1))

            # Store original target *before* repeat
            orig_targ_np = ak.to_numpy(targ_data)
            if orig_targ_np.ndim == 1:
                orig_targ_np = orig_targ_np.reshape(-1, 1)

            # Broadcast target using numpy.repeat: (N, 1) -> (N*var, 1)
            targ_flat = np.repeat(orig_targ_np, counts_np, axis=0)

        else:
            # Standard Scenario: Flatten all var dims automatically
            logger.debug(
                f"Running standard flattening scenario for shapes "
                f"{feat_shape} and {targ_shape}"
            )

            if feat_data is not None:
                feat_flat = ak.to_numpy(ak_flatten_variable_dims(feat_data, axis=None))
                if feat_flat.ndim == 1:
                    feat_flat = feat_flat.reshape(-1, 1)

            if targ_data is not None:
                targ_flat = ak.to_numpy(ak_flatten_variable_dims(targ_data, axis=None))
                if targ_flat.ndim == 1:
                    targ_flat = targ_flat.reshape(-1, 1)

        # --- 3. Final Validation ---
        if feat_flat is not None and targ_flat is not None:
            if feat_flat.shape[0] != targ_flat.shape[0]:
                raise ValueError(
                    f"Time dimension mismatch after flattening: "
                    f"Features have {feat_flat.shape[0]} samples (from {feat_shape}), "
                    f"Targets have {targ_flat.shape[0]} samples (from {targ_shape}). "
                    f"Scenario 1 was {is_scenario_1}."
                )

        return feat_flat, targ_flat, orig_targ_np

    def _save_plot(self, fig: plt.Figure, unit_id_str: str, regime: str):
        """Creates directory and saves the plot."""
        regime_path = os.path.join(self.saving_path, regime)
        os.makedirs(regime_path, exist_ok=True)

        filename = f"{unit_id_str}.png"
        file_path = os.path.join(regime_path, filename)

        try:
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            logger.debug(f"Saved plot to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save plot to {file_path}: {e}")
        finally:
            plt.close(fig)  # Close figure to free memory

    def _save_original_target_plot(self, orig_targ_data: np.ndarray, unit_id_str: str):
        """Saves a separate plot of the original (N, 1) target."""
        n_samples, n_features = orig_targ_data.shape
        x_axis = np.arange(n_samples)

        # Use a single plot or subplots if multiple target features
        ncols = 2
        nrows = max(1, math.ceil(n_features / ncols))
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 3), squeeze=False
        )
        axes = axes.flatten()

        for idx in range(nrows * ncols):
            ax = axes[idx]
            if idx < n_features:
                ax.plot(
                    x_axis, orig_targ_data[:, idx], lw=1, color="#2ca02c", alpha=0.9
                )
                ax.set_title(f"Original Target {idx}", pad=4)
                ax.set_xlabel("Sample (N)")
                ax.set_ylabel("Target Value")
            else:
                ax.axis("off")

        fig.suptitle(
            f"Original Target Plot for Unit: {unit_id_str}", fontsize=14, y=1.02
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        self._save_plot(fig, unit_id_str, "original_targets")

    @override
    def transform_data(
        self, data: NamedTransformInput, metadata: Dict
    ) -> NamedTransformInput:
        """
        Generates and saves the plot(s), then returns the data unmodified.
        """
        unit_id_str = self._get_unit_id(
            {self.unit_key: data.get(self.unit_key, "unknown_unit")}
            if self.unit_key
            else {}
        )

        mode = metadata.get("mode", None)
        unit_id_str = f"{mode}_{unit_id_str}"

        try:
            # --- 1. Prepare Data ---
            feat_data, targ_data, orig_targ_data = self._prepare_data(data)

            # --- 2. Save Original Target Plot (if it exists) ---
            if orig_targ_data is not None:

                try:
                    self._save_original_target_plot(orig_targ_data, unit_id_str)
                except Exception as e:
                    logger.error(
                        f"Failed to plot original target for unit '{unit_id_str}': {e}"
                    )

            # --- 3. Save Main (Flattened) Plot ---
            n_features = 0
            n_samples = 0

            if self.regime == "features_and_targets":
                n_features = feat_data.shape[1]
                n_samples = feat_data.shape[0]
            elif self.regime == "features_only":
                n_features = feat_data.shape[1]
                n_samples = feat_data.shape[0]
            elif self.regime == "targets_only":
                n_features = targ_data.shape[1]
                n_samples = targ_data.shape[0]

            ncols = 2
            nrows = max(1, math.ceil(n_features / ncols))
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 3), squeeze=False
            )
            axes = axes.flatten()
            x_axis = np.arange(n_samples)

            for idx in range(nrows * ncols):
                ax = axes[idx]
                if idx < n_features:
                    if self.regime == "features_and_targets":
                        y_feat = feat_data[:, idx]
                        ax.plot(
                            x_axis,
                            y_feat,
                            lw=1,
                            color="#1f77b4",
                            alpha=0.9,
                            label="Feature",
                        )
                        ax.set_ylabel(f"Feature {idx} Value", color="#1f77b4")
                        ax.tick_params(axis="y", labelcolor="#1f77b4")

                        ax_twin = ax.twinx()
                        y_targ = (
                            targ_data[:, 0]
                            if targ_data.shape[1] == 1
                            else targ_data[:, idx % targ_data.shape[1]]
                        )
                        ax_twin.plot(
                            x_axis,
                            y_targ,
                            lw=1,
                            color="#d62728",
                            linestyle="--",
                            alpha=0.8,
                            label="Target",
                        )
                        ax_twin.set_ylabel("Target Value", color="#d62728")
                        ax_twin.tick_params(axis="y", labelcolor="#d62728")
                        ax.set_title(f"Feature {idx} vs. Target", pad=4)

                    elif self.regime == "features_only":
                        y_feat = feat_data[:, idx]
                        ax.plot(x_axis, y_feat, lw=1, color="#1f77b4", alpha=0.9)
                        ax.set_ylabel("Value")
                        ax.set_title(f"Feature {idx}", pad=4)

                    elif self.regime == "targets_only":
                        y_targ = targ_data[:, idx]
                        ax.plot(x_axis, y_targ, lw=1, color="#d62728", alpha=0.9)
                        ax.set_ylabel("Value")
                        ax.set_title(f"Target {idx}", pad=4)

                    ax.set_xlabel("Sample (Flattened Time)")
                else:
                    ax.axis("off")

            fig.suptitle(
                f"{self.regime.replace('_', ' ').title()} Plot for Unit: {unit_id_str}",
                fontsize=14,
                y=1.02,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.98])

            self._save_plot(fig, unit_id_str, self.regime)

        except Exception as e:
            logger.error(
                f"Failed to plot features for unit '{unit_id_str}'. Error: {e}",
                exc_info=True,
            )
            pass

        # Return the original data unmodified
        return data

    def __call__(self, data: Dict, metadata: Dict) -> Dict[str, np.ndarray]:
        return self.transform_data(data, metadata)
