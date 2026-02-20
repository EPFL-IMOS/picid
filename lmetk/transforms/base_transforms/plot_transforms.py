from typing import Any
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform

import numpy as _np
import math as _math
import matplotlib.pyplot as _plt

_plt.switch_backend("Agg")

from lmetk.transforms.base.multisource import ConcatFitAndPerSegmentTransformMixin


class PlotFeatures(DenseTransform, ConcatFitAndPerSegmentTransformMixin):
    """
    An adapter for sklearn.preprocessing.StandardScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self, unit_key: str = "unit_id"):
        self.unit_key = unit_key
        super().__init__()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        pass

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
        """Transforms a single data segment using the internal StandardScaler."""
        keys = list(data.keys())
        assert len(keys) == 2, "PlotFeatures only supports two key data_segment."

        np_data = data[keys[0]]

        # Ensure numpy array and 2D shape (n_samples, n_features)
        np_data = _np.asarray(np_data)
        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)

        n_samples, n_features = np_data.shape
        ncols = 2
        nrows = max(1, _math.ceil(n_features / ncols))

        # Publication-ready rc settings
        _plt.rcParams.update(
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

        fig, axes = _plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 2.5), squeeze=False
        )
        axes = axes.flatten()

        x = _np.arange(n_samples)
        for idx in range(nrows * ncols):
            ax = axes[idx]
            if idx < n_features:
                y = np_data[:, idx]
                ax.plot(x, y, lw=1, color="#1f77b4", alpha=0.9)
                ax.set_title(f"Feature {idx + 1}", pad=4)
                ax.set_xlabel("Sample")
                ax.set_ylabel("Value")
                # annotate basic stats
                mean = _np.nanmean(y)
                std = _np.nanstd(y)
                ax.text(
                    0.98,
                    0.90,
                    f"μ={mean:.2f}\nσ={std:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                )
            else:
                ax.axis("off")

        fig.suptitle("Feature traces", fontsize=12, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        title = ""
        unit_id_str = "unknown_unit"

        unit_id = data.get(self.unit_key, "unknown_unit")
        if isinstance(unit_id, (tuple, list)):
            unit_id_str = "_".join(map(str, unit_id))
        else:
            unit_id_str = str(unit_id).replace(" ", "_")
        title = unit_id_str

        filename = f"{title}.png"
        fig.savefig(f"plot_dir/{filename}", dpi=300)

        return data
