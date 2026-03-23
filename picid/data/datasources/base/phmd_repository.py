"""Shared PHMD repository helpers.

This module isolates the PHMD-specific dataset lookup and cache-name logic from
the loader classes so the public datasource implementations can focus on
loader lifecycle and per-unit conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any

from phmd import datasets


@dataclass(slots=True)
class PHMDFoldBundle:
    """Resolved PHMD task folds and metadata.

    Attributes
    ----------
    task_folds : dict[str, Any]
        Mapping from task name to the selected fold payload.
    meta_data : Any
        Metadata attached to the active task in the PHMD dataset.
    """

    task_folds: dict[str, Any]
    meta_data: Any


class PHMDRepository:
    """Repository for PHMD dataset access and cache naming.

    Parameters
    ----------
    data_name : str
        PHMD dataset name used to resolve the dataset family.
    cache_dir : str
        Local cache directory where PHMD stores downloaded assets.
    fold : int
        Fold index used to select the active task payload.
    task_mode : str
        Name of the primary task requested by the datasource.
    auxiliary_tasks : list[str] | None
        Optional auxiliary tasks resolved alongside the main task.
    """

    def __init__(
        self,
        *,
        data_name: str,
        cache_dir: str,
        fold: int,
        task_mode: str,
        auxiliary_tasks: list[str] | None = None,
    ) -> None:
        self.data_name = data_name
        self.cache_dir = str(Path(cache_dir).expanduser())
        self.fold = fold
        self.task_mode = task_mode
        self.auxiliary_tasks = list(auxiliary_tasks or [])

        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)

    def load_task_bundle(self, task_names: list[str]) -> PHMDFoldBundle:
        """Resolve the requested PHMD task folds.

        Parameters
        ----------
        task_names : list[str]
            Task names to retrieve from the PHMD dataset. The primary task
            should be included so its metadata can be captured.

        Returns
        -------
        PHMDFoldBundle
            The selected fold payloads plus metadata from the primary task.
        """

        ds = datasets.Dataset(self.data_name, cache_dir=self.cache_dir)

        task_folds: dict[str, Any] = {}
        meta_data: Any = {}
        for task_name in task_names:
            task = ds.get_task(task_name)
            task_folds[task_name] = task[self.fold]
            if task_name == self.task_mode:
                # Keep the metadata from the active task only once so the
                # loader can enrich it with per-unit provenance later on.
                meta_data = task.meta

        return PHMDFoldBundle(task_folds=task_folds, meta_data=meta_data)

    def get_cache_file_name(self, data_type: str) -> str:
        """Return a stable cache filename for a PHMD payload.

        Parameters
        ----------
        data_type : str
            Logical payload type, such as ``"train"`` or ``"metadata"``.

        Returns
        -------
        str
            Cache filename derived from the task mode, fold, auxiliary tasks,
            dataset name, and payload type.
        """

        filename_parts = [
            f"taskmode_{self.task_mode}",
            f"fold_{self.fold}",
            f"aux_{'_'.join(self.auxiliary_tasks) if self.auxiliary_tasks else 'none'}",
            f"dataset_{self.data_name}",
            f"type_{data_type}",
        ]
        return "_".join(filename_parts) + ".pkl"

    def retrieve_cached_task_data(self, path: Path) -> Any:
        """Load a previously cached PHMD payload from disk.

        Parameters
        ----------
        path : pathlib.Path
            Cache file to read.

        Returns
        -------
        Any
            Unpickled cached payload.

        Raises
        ------
        FileNotFoundError
            If the cache file does not exist.
        """

        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")
        with open(path, "rb") as file_obj:
            return pickle.load(file_obj)
