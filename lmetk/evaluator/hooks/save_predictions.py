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

    This hook ensures that 3D model outputs are saved with a consistent schema.
    It resolves dimension size conflicts (e.g., in classification) by dynamically
    renaming target dimensions if they differ from prediction dimensions.
    """

    def __init__(self, dims: Optional[List[str]] = None):
        """
        Args:
            dims: Explicit dimension names for 3D tensors.
                 Required to be set, usually: ["sample", "time", "feature"].
                 If not provided, a ValueError is raised during formatting.
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
        # 1. Guard: Only run if the evaluator is configured to save
        if not getattr(evaluator, "save_predictions", False):
            return

        # 2. Guard: Ensure the buffer has data to save
        data = evaluator.buffer.get_all()
        if not data or data.get("preds") is None:
            return

        # 3. Format data into xarray-compatible structure
        try:
            xr_data = self._format_xarray(data, evaluator)
        except ValueError as e:
            logger.error(f"SavePredictionsHook failed: {e}")
            raise

        # 4. Resolve output path
        p_dir = getattr(evaluator.paths, "eval_details", None)
        if not p_dir:
            logger.warning(
                "SavePredictionsHook: 'eval_details' path missing. Skipping save."
            )
            return

        out_path = Path(p_dir) / mode
        out_path.mkdir(parents=True, exist_ok=True)

        save_file = out_path / "predictions.nc"

        # 5. Build Dataset and write to disk
        # Conflicts between pred/target feature sizes are handled in _format_xarray
        ds = xr.Dataset(xr_data)
        ds.to_netcdf(save_file)
        logger.info(f"Saved NetCDF predictions to {save_file}")

    def _format_xarray(
        self, data: Dict[str, np.ndarray], evaluator: Any
    ) -> Dict[str, Any]:
        """
        Maps raw buffer arrays to xarray-compatible tuples with standardized dims.
        """
        if self.dims is None:
            raise ValueError(
                "SavePredictionsHook: No dimension names provided during initialization. "
                "Please initialize with dims=['sample', 'time', 'feature'] or similar."
            )

        if len(self.dims) != 3:
            raise ValueError(
                f"SavePredictionsHook: Provided dims {self.dims} must have exactly 3 elements."
            )

        res = {}
        # Shapes used for conflict detection (e.g. logit size vs label size)
        p_shape = data["preds"].shape

        # Core data variables
        for key in ["preds", "targets", "norm_preds", "norm_targets"]:
            val = data.get(key)
            if val is None:
                continue

            # Mapping internal 'norm_x' prefix to external 'x_normalized' suffix
            if key.startswith("norm_"):
                base_name = key.replace("norm_", "")
                store_key = f"{base_name}_normalized"
            else:
                store_key = key

            if val.ndim == 3:
                current_dims = list(self.dims)

                # Logic: In classification, preds might be (N, T, 5) while targets are (N, T, 1).
                # Xarray requires unique dimension names if lengths differ.
                if "target" in key and val.shape[2] != p_shape[2]:
                    current_dims[2] = f"{current_dims[2]}_label"

                res[store_key] = (current_dims, val)
            else:
                # Fallback for non-3D tensors (unlikely for preds/targets in this framework)
                fallback_dims = [f"dim_{i}" for i in range(val.ndim)]
                res[store_key] = (fallback_dims, val)

        # Metadata: unit_ids
        if data.get("unit_ids") is not None:
            u_ids = data["unit_ids"]
            # unit_ids always share the 'sample' dimension with predictions
            u_dims = [self.dims[0]] + [f"unit_dim_{i}" for i in range(1, u_ids.ndim)]
            res["unit_ids"] = (u_dims, u_ids)

        return res
