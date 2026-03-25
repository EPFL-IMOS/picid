"""File-based synthetic loader for pipeline snapshot tests. Reads pre-generated npz files."""

from pathlib import Path
from typing import Any

import numpy as np

from picid.data.datasources.base.single_source_loader import SingleSourceLoader
from picid.data.split_strategies import TimeSplitter


class SyntheticFromFileLoader(SingleSourceLoader):
    """Loads synthetic data from npz files. Keys: task_type-specific."""

    def __init__(
        self,
        data_path: str,
        task_mode: str,
        data_name: str = "snapshot",
        data_splitter: Any = None,
        **kwargs,
    ):
        if data_splitter is None:
            create_splits = {
                "rul": ["features", "timestamps", "rul", "unit_id"],
                "fault": ["features", "timestamps", "target"],
                "anomaly_detection": ["features", "timestamps", "anomaly_detection"],
                "forecasting": ["features", "timestamps", "target"],
            }[task_mode]
            data_splitter = TimeSplitter(
                train=0.5,
                val=0.25,
                test=None,
                seq_len=10,
                pred_len=1,
                create_splits_for=create_splits,
            )
        super().__init__(
            data_splitter=data_splitter,
            data_name=data_name,
            task_mode=task_mode,
            **kwargs,
        )
        self._data_path = Path(data_path)

    def _load_data(self) -> dict:
        data = np.load(self._data_path)
        out = {k: data[k] for k in data.files}
        n = len(out["features"])
        out["unit_id"] = np.zeros((n, 1), dtype=np.int64)
        return out
