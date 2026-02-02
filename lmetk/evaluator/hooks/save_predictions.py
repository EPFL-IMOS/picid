from pathlib import Path
from typing import Any, Dict

import numpy as np
import xarray as xr

from lmetk.evaluator.hooks.base import BaseEvalHook


class SavePredictionsHook(BaseEvalHook):
    """Handles NetCDF saving with task-specific dimension naming."""

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
        if not data:
            return

        xr_data = self._format_xarray(data, evaluator)
        p_dir = getattr(evaluator.paths, "eval_details", None)

        if p_dir:
            out_path = Path(p_dir) / mode
            out_path.mkdir(parents=True, exist_ok=True)
            xr.Dataset(xr_data).to_netcdf(out_path / "predictions.nc")

    def _format_xarray(
        self, data: Dict[str, np.ndarray], evaluator: Any
    ) -> Dict[str, Any]:
        p = data["preds"]
        task_type = getattr(evaluator, "task_type", "")

        if task_type == "classification" and p.ndim > 2:
            d_names = ["sample", "time", "class_logits"]
            t_names = ["sample", "time", "class_label"]
        else:
            d_names = ["sample"] + [f"dim{i}" for i in range(1, p.ndim)]
            t_names = d_names

        res = {"preds": (d_names, p), "targets": (t_names, data["targets"])}

        if data.get("norm_preds") is not None:
            res["preds_normalized"] = (d_names, data["norm_preds"])
            res["targets_normalized"] = (t_names, data["norm_targets"])

        if data.get("unit_ids") is not None:
            u = data["unit_ids"]
            u_dims = ["sample"] + [f"unit_dim{i}" for i in range(1, u.ndim)]
            res["unit_ids"] = (u_dims, u)

        return res
