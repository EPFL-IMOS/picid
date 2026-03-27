"""
Shared loader for PHMD-backed multi-source datasets.

The PHMD datasets follow a common pattern: each task is stored in a cached
dataset object, units are grouped by a unit identifier column, and each unit is
converted into a nested train/val/test payload. This module delegates the
dataset lookup and split assembly to dedicated helpers so concrete PHMD loaders
only implement dataset-specific column selection and per-unit conversion.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from picid.data.datasources.base.contracts import LoaderState
from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)
from picid.data.datasources.base.phmd_assembly import build_split_payloads
from picid.data.datasources.base.phmd_payload_cache import (
    load_phmd_payload_cache,
    save_phmd_payload_cache,
)
from picid.data.datasources.base.phmd_repository import PHMDRepository
from picid.data.datasources.utils import convert_outer_list_to_inner

logger = logging.getLogger(__name__)

__all__ = ["PHMDMultiSourceLoader", "convert_outer_list_to_inner"]


class PHMDMultiSourceLoader(PredefinedSplitLoaderBase, ABC):
    """
    Provide a base class for PHMD loaders that expose one record per unit.

    Concrete subclasses provide the feature, target, and unit-column mapping,
    while this base class handles PHMD dataset retrieval, unit-wise iteration,
    and assembly of the final split container.

    Parameters
    ----------
    **kwargs
        Loader configuration forwarded to :class:`PredefinedSplitLoaderBase`
        together with PHMD-specific arguments such as ``fold`` and
        ``cache_dir``. Optional ``payload_cache_path`` (``str`` or
        :class:`pathlib.Path`): after the first successful :meth:`load_data`,
        writes processed ``data_dict`` and ``meta_data`` with
        :meth:`get_cache_fingerprint` for validation; later runs restore from
        disk when the fingerprint matches, skipping PHMD reads and per-unit
        processing. Intended for notebooks and analysis. The path is omitted from
        the fingerprint so moving the cache file does not change preprocessor
        cache keys.
    """

    def __init__(self, **kwargs):
        """
        Initialise the PHMD loader and prepare the local cache directory.

        Parameters
        ----------
        **kwargs
            Loader configuration including ``fold``, ``cache_dir``, and
            optional ``payload_cache_path`` (see class docstring).
        """
        super().__init__(**kwargs)
        self.fold: int = kwargs["fold"]
        self.dataset_fold: str = kwargs.get("dataset_fold", None)
        self.cache_dir: str = kwargs["cache_dir"]
        self.auxiliary_tasks: list = kwargs.get("auxiliary_tasks", [])
        self.use_ragged: bool = kwargs.get("use_ragged", False)

        cache_path = Path(self.cache_dir).expanduser()
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = str(cache_path)
        self.meta_data = {}
        self._repository = PHMDRepository(
            data_name=self.data_name,
            cache_dir=self.cache_dir,
            fold=self.fold,
            task_mode=self.task_mode,
            auxiliary_tasks=self.auxiliary_tasks,
        )
        raw_pc = kwargs.get("payload_cache_path")
        self._payload_cache_path: str | None = (
            str(Path(raw_pc).expanduser()) if raw_pc is not None else None
        )

    def _build_cache_config(self):
        """
        Return cache fingerprint without ``payload_cache_path`` (I/O only).

        Returns
        -------
        dict
            Serialisable fingerprint for :meth:`get_cache_fingerprint`.
        """
        fp = super()._build_cache_config()
        fp.pop("payload_cache_path", None)
        return fp

    def load_data(self) -> None:
        """Load split payload, optionally restoring from ``payload_cache_path``."""
        if self._payload_cache_path:
            restored = load_phmd_payload_cache(
                self._payload_cache_path,
                self.get_cache_fingerprint(),
            )
            if restored is not None:
                self.data_dict = restored["data_dict"]
                self.meta_data = restored["meta_data"]
                self._is_loaded = True
                self._is_splitted = True
                self._state = LoaderState.SPLIT
                return

        super().load_data()

        if self._payload_cache_path:
            assert self.data_dict is not None
            save_phmd_payload_cache(
                self._payload_cache_path,
                self.get_cache_fingerprint(),
                self.data_dict,
                self.meta_data,
            )

    def __repr__(self) -> str:
        """Return a compact debug representation of the loader state."""
        if self._is_loaded:
            return f"MultiSource(sources={len(getattr(self, 'data_source_dict', {}))} units)"
        return "MultiSource(unloaded)"

    def get_multisource_data_splitter(self):
        """
        Return the configured multisource splitter, if any.

        Returns
        -------
        object
            The configured multisource splitter or ``None``.
        """
        return self.multisource_data_splitter

    def _init_phmd_dataset(self, task_names):
        """
        Load the PHMD dataset and resolve the requested task folds.

        Parameters
        ----------
        task_names : list[str]
            Tasks to resolve from the PHMD dataset, including the main task and
            any auxiliary tasks required for preprocessing.

        Returns
        -------
        dict[str, Any]
            Mapping from task name to the selected fold payload.
        """
        bundle = self._repository.load_task_bundle(task_names)
        self.meta_data = bundle.meta_data
        return bundle.task_folds

    def _retrieve_cached_task_data(self, path: Path):
        """
        Load a cached task payload from disk.

        Parameters
        ----------
        path : pathlib.Path
            Cache file path to load.

        Returns
        -------
        object
            Deserialized cached payload.
        """
        return self._repository.retrieve_cached_task_data(path)

    def _get_cache_file_name(self, data_type) -> str:
        """
        Return a stable cache filename for the requested payload type.

        Parameters
        ----------
        data_type : str
            Split or payload type to encode into the cache filename.

        Returns
        -------
        str
            Stable cache filename for the requested payload type.
        """
        return self._repository.get_cache_file_name(data_type)

    def _load_data(self):
        """
        Load and convert the PHMD dataset into split-aware unit payloads.

        Returns
        -------
        dict[str, dict[str, list]]
            Nested split payload with per-unit feature, target, and metadata
            lists for ``train``, ``val``, and ``test``.
        """
        self.meta_data = {}
        tasks_fold_dict = self._init_phmd_dataset(
            self.auxiliary_tasks + [self.task_mode]
        )

        main_fold = tasks_fold_dict[self.task_mode]
        auxiliary_tasks_fold_dict = tasks_fold_dict.copy()
        auxiliary_tasks_fold_dict.pop(self.task_mode, None)

        main_fold = self._preprocess_fold(main_fold, auxiliary_tasks_fold_dict)

        # Modify the splits if needed by source:
        if self.dataset_fold is not None:
            raise NotImplementedError(
                "dataset_fold-specific PHMD splitting is not implemented in PHMDMultiSourceLoader."
            )

        out_dict, extra_meta = build_split_payloads(
            main_fold,
            unit_column=self._get_unit_column(),
            feature_columns=self._get_features_columns(),
            target_column=self._get_target_column(),
            process_unit=self._process_unit,
            debug_subset_range=self.debug_subset_range,
        )
        if extra_meta:
            self.meta_data.update(extra_meta)

        features_splits = out_dict.get("features", {})
        logger.info(f"Number of train units: {len(features_splits.get('train', []))}")
        logger.info(f"Number of val units: {len(features_splits.get('val', []))}")
        logger.info(f"Number of test units: {len(features_splits.get('test', []))}")
        return out_dict

    @abstractmethod
    def _process_unit(self, df_unit, unit_name, features_col, target_col):
        """
        Convert one unit-level dataframe into the final payload shape.

        Parameters
        ----------
        df_unit : pandas.DataFrame
            Unit-level dataframe to convert.
        unit_name : str
            Unit identifier used for metadata and diagnostics.
        features_col : list[str]
            Feature columns selected for the unit payload.
        target_col : str
            Target column selected for the active task mode.
        """
        raise NotImplementedError("Subclasses must implement _process_unit")

    @abstractmethod
    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):
        """
        Prepare the task folds before unit-level conversion.

        Parameters
        ----------
        main_fold : dict[str, pandas.DataFrame]
            Fold data for the primary task.
        auxiliary_tasks_fold_dict : dict[str, pandas.DataFrame]
            Fold data for auxiliary tasks used to enrich the primary task.
        """
        raise NotImplementedError("Subclasses must implement _rename_columns")

    @abstractmethod
    def _get_features_columns(self) -> list[str]:
        """Return the feature columns used to build each unit payload."""
        raise NotImplementedError("Subclasses must implement _get_features_columns")

    @abstractmethod
    def _get_target_column(self) -> str:
        """Return the target column used for the active task mode."""
        raise NotImplementedError("Subclasses must implement _get_target_column")

    @abstractmethod
    def _get_unit_column(self) -> str:
        """Return the column that identifies each physical unit."""
        raise NotImplementedError("Subclasses must implement _get_unit_column")
