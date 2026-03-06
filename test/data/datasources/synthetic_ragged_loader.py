"""Synthetic loader that produces ragged (variable-length) data mimicking unibo/pronostia/xjtu."""
from __future__ import annotations

from typing import Any, Dict, List

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


class SyntheticRaggedLoader(SingleSourceLoader):
    """
    Produces ragged data: list of ak.Array per split, variable-length sequences per unit.
    Mimics structure of unibo (battery), pronostia, xjtu (bearings).
    """

    def __init__(
        self,
        seed: int = 42,
        n_units: int = 3,
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
        self._seed = seed
        self._n_units = n_units

    def split_data(self) -> None:
        """Assign units to train/val/test (unit 0->train, 1->val, 2->test)."""
        if self._is_splitted:
            return
        assert self._is_loaded, "Data must be loaded before splitting."
        splits = ["train", "val", "test"]
        for key in ["features", "rul", "unit_id"]:
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
            raise ImportError("awkward required for SyntheticRaggedLoader")
        rng = np.random.default_rng(self._seed)
        # Per unit: variable-length sequences (n_cycles varies)
        lengths = [50 + int(rng.integers(0, 30)) for _ in range(self._n_units)]
        features_list: List[ak.Array] = []
        rul_list: List[ak.Array] = []
        unit_id_list: List[np.ndarray] = []
        for i, L in enumerate(lengths):
            seq_len = 20 + int(rng.integers(0, 10))
            feat = rng.standard_normal((L, seq_len, 3)).astype(np.float32)
            features_list.append(ak.Array(feat))
            rul_list.append(ak.Array(np.linspace(1.0, 0.0, L).astype(np.float32)))
            unit_id_list.append(np.full((L, 1), i, dtype=np.int64))
        return {
            "features": features_list,
            "rul": rul_list,
            "unit_id": unit_id_list,
        }
