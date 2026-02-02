import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from lmetk.evaluator.hooks.base import BaseEvalHook

logger = logging.getLogger(__name__)


class SavePredictionsHook(BaseEvalHook):
    """
    Handles NetCDF saving with standardized dimension naming.
    """

    def __init__(self, dims: Optional[List[str]] = None):
        """
        Args:
            dims: Explicit dimension names for 3D tensors.
                 Example: ["sample", "time", "feature"].
        """
        self.dims = dims

    def on_compute_end(
        self,
        results: Dict[str, float],
        evaluator: Any,
        mode: str,
        epoch: int,
        step: int,
    ) -> None:
        if not getattr(evaluator, "save_predictions", False):
            return

        data = evaluator.buffer.get_all()
        if not data or data.get("preds") is None:
            return

        try:
            xr_data = self._format_xarray(data, evaluator)
        except ValueError as e:
            logger.error(f"SavePredictionsHook failed: {e}")
            raise

        p_dir = getattr(evaluator.paths, "eval_details", None)
        if not p_dir:
            return

        out_path = Path(p_dir) / mode
        out_path.mkdir(parents=True, exist_ok=True)

        save_file = out_path / "predictions.nc"
        # The xr.Dataset constructor will now succeed because dim sizes match or names differ
        xr.Dataset(xr_data).to_netcdf(save_file)

    def _format_xarray(
        self, data: Dict[str, np.ndarray], evaluator: Any
    ) -> Dict[str, Any]:
        if self.dims is None:
            raise ValueError(
                "SavePredictionsHook: No dimension names provided during initialization."
            )

        if len(self.dims) != 3:
            raise ValueError(
                f"SavePredictionsHook: Provided dims {self.dims} must have 3 elements."
            )

        res = {}
        # We need to detect if targets and preds have different feature sizes (common in classification)
        p_shape = data["preds"].shape
        t_shape = data["targets"].shape

        # Primary mapping
        for key in ["preds", "targets", "norm_preds", "norm_targets"]:
            val = data.get(key)
            if val is None:
                continue

            store_key = key if "norm" not in key else f"{key.split('_')[0]}_normalized"

            if val.ndim == 3:
                # Logic: If this is a target variable and its feature dim (index 2)
                # is different from the prediction feature dim, we rename the dimension
                # to avoid Xarray size conflicts.
                current_dims = list(self.dims)
                if "target" in key and val.shape[2] != p_shape[2]:
                    current_dims[2] = f"{current_dims[2]}_label"

                res[store_key] = (current_dims, val)
            else:
                fallback_dims = [f"dim_{i}" for i in range(val.ndim)]
                res[store_key] = (fallback_dims, val)

        if data.get("unit_ids") is not None:
            u_ids = data["unit_ids"]
            u_dims = [self.dims[0]] + [f"unit_dim_{i}" for i in range(1, u_ids.ndim)]
            res["unit_ids"] = (u_dims, u_ids)

        return res
