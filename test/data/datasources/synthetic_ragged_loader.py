"""File-based synthetic ragged loader for pipeline snapshot tests. Reads pre-generated fixtures."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import awkward as ak
except ImportError:
    ak = None

from picid.data.datasources.base.single_source_loader import SingleSourceLoader


def _length_of_array(arr: Any) -> int:
    """Get first-dimension length for numpy or ak.Array."""
    try:
        return len(arr)
    except TypeError:
        pass
    sh = getattr(arr, "shape", None)
    if sh is not None and len(sh) > 0:
        return int(sh[0])
    return len(arr)


class SyntheticRaggedFromFileLoader(SingleSourceLoader):
    """
    Loads pre-generated ragged data from pickle. Deterministic across machines.
    Mimics structure of unibo (battery), pronostia, xjtu (bearings).
    """

    def __init__(
        self,
        data_path: str,
        data_name: str = "synthetic_ragged",
        task_mode: str = "rul",
        **kwargs: Any,
    ):
        super().__init__(
            data_splitter=None,
            data_name=data_name,
            task_mode=task_mode,
            **kwargs,
        )
        self._data_path = Path(data_path)

    def split_data(self) -> None:
        """Assign units to train/val/test (unit 0->train, 1->val, 2->test)."""
        if self._is_splitted:
            return
        assert self._is_loaded, "Data must be loaded before splitting."
        splits = ["train", "val", "test"]
        for key in ["features", "rul", "target", "unit_id"]:
            data = self.data_dict[key]
            if not isinstance(data, list):
                raise ValueError(f"Expected list for {key}, got {type(data)}")
            n_units = len(data)
            splitted = {s: None for s in splits}
            masks: Dict[str, np.ndarray] = {}
            for i, split_name in enumerate(splits):
                unit_idx = min(i, n_units - 1)
                arr = data[unit_idx]
                splitted[split_name] = arr
                masks[split_name] = np.ones(_length_of_array(arr), dtype=bool)
            if "split_masks" not in self.data_dict:
                self.data_dict["split_masks"] = masks
            self.data_dict[key] = splitted
        self._is_splitted = True

    def _load_data(self) -> Dict[str, Any]:
        if ak is None:
            raise ImportError("awkward required for SyntheticRaggedFromFileLoader")
        with open(self._data_path, "rb") as f:
            data = pickle.load(f)
        # Convert to ak.Array for transform compatibility
        return {
            "features": [ak.Array(a) for a in data["features"]],
            "rul": [ak.Array(a) for a in data["rul"]],
            "target": [ak.Array(a) for a in data["target"]],
            "unit_id": data["unit_id"],
        }
