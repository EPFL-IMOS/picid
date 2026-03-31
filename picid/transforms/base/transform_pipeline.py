"""
transform_pipeline.py
=====================

Library-facing orchestration layer for chaining DataTransform objects without
requiring Hydra, OmegaConf, or ConfigTransformManager.

Two public classes:

    TransformSequenceProtocol
        Structural protocol (typing.Protocol) that both TransformPipeline and
        ConfigTransformManager satisfy.  PreProcessor and the pipeline runners
        depend only on this protocol — not on either concrete class.

    TransformPipeline
        Lightweight sequential transform runner for library users.  Accepts a
        list of DataTransform objects constructed directly in Python.  Supports
        the same three-tier caching as the framework path (loaded_data →
        boundary → preprocessed) when passed to PreProcessor.pipeline().

Library usage — no caching:
-----------------------------
    from picid.transforms.base.transform_pipeline import TransformPipeline
    from picid.transforms.base.data_transform import DataTransform

    pipeline = TransformPipeline([
        DataTransform("normalize", scaler,  {"apply_to": "features", "fit_on": "train"}),
        DataTransform("window",    windower, {"apply_to": "features"}),
        DataTransform("scale_y",   scaler2, {"apply_to": "target",   "fit_on": "train"}),
    ])
    data = pipeline.run(data)

Library usage — with caching (same three-tier system as framework):
--------------------------------------------------------------------
    # Mark a heavy transform as a cache boundary:
    DataTransform("window", windower, {"apply_to": "features", "cache_point": True})

    preprocessor = PreProcessor(datasource=datasource, transforms=pipeline)
    data = preprocessor.pipeline(
        data_cache_path=".cache",
        data_library_part_path="./picid/data/datasources",
        transform_library_part_path="./picid/transforms",
        cache_preprocessed=True,
    )

Cache key computation for TransformPipeline
--------------------------------------------
ConfigTransformManager derives cache keys from its OmegaConf DictConfig (which
captures every constructor argument for every transform).  TransformPipeline has
no config file, so it derives an equivalent serialisable dict from each
DataTransform's name, class, and metadata dict.  The resulting structure is
compatible with compute_cache_key and is deterministic for the same pipeline
definition.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Protocol, runtime_checkable

from picid.transforms.base.data_transform import DataTransform
from picid.data.data_objects import SplitDatasetContainer
from picid.utils.hash_utils import ensure_serializable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — what the pipeline runners actually need
# ---------------------------------------------------------------------------


@runtime_checkable
class TransformSequenceProtocol(Protocol):
    """
    Structural protocol satisfied by both ConfigTransformManager and
    TransformPipeline.  PreProcessor and the pipeline runners depend only
    on this interface.

    Any object that exposes these five members can be passed as the
    `transforms` argument to PreProcessor.
    """

    @property
    def config(self) -> Any:
        """
        Return a serializable representation of the full transform config.

        Returns
        -------
        Any
            Stable transform configuration used for cache key computation.
        """
        ...

    def get_transforms(self) -> OrderedDict[str, DataTransform]:
        """
        Return all transforms in pipeline order.

        Returns
        -------
        OrderedDict
            Ordered mapping of transform names to `DataTransform` objects.
        """
        ...

    def get_cache_point_names(self) -> List[str]:
        """
        Return transform names that are cache boundaries.

        Returns
        -------
        list of str
            Transform names marked as cache points, in pipeline order.
        """
        ...

    def get_transform_names_after(self, transform_name: str) -> List[str]:
        """
        Return transform names that come after a named transform.

        Parameters
        ----------
        transform_name : str
            Boundary transform name.

        Returns
        -------
        list of str
            Transform names after the provided boundary.
        """
        ...

    def get_config_up_to_and_including(self, transform_name: str) -> Dict:
        """
        Return a serializable config slice up to a named transform.

        Parameters
        ----------
        transform_name : str
            Last transform name to include.

        Returns
        -------
        dict
            Serializable configuration slice used for boundary cache keys.
        """
        ...


# ---------------------------------------------------------------------------
# TransformPipeline
# ---------------------------------------------------------------------------


class TransformPipeline:
    """
    Sequential transform runner for library users.

    Accepts a list of DataTransform objects and satisfies
    TransformSequenceProtocol so it can be passed directly to PreProcessor
    and benefit from the same three-tier caching as the framework path.

    Parameters
    ----------
    transforms : list of DataTransform
        Ordered list of transforms to apply.  Transform names must be unique.

    Raises
    ------
    ValueError
        If any element is not a DataTransform, or if names are not unique.
    """

    def __init__(self, transforms: List[DataTransform]):
        if not all(isinstance(t, DataTransform) for t in transforms):
            raise ValueError(
                "All elements of TransformPipeline must be DataTransform instances."
            )
        names = [t.transform_name for t in transforms]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"TransformPipeline: transform names must be unique. "
                f"Duplicates: {sorted(duplicates)}"
            )

        self._transforms: OrderedDict[str, DataTransform] = OrderedDict(
            (t.transform_name, t) for t in transforms
        )
        # Build and cache the serialisable config on construction so it is
        # stable and cheap to access repeatedly (e.g. for cache key computation).
        self._config: Dict[str, Any] = self._build_config(self._transforms)

    # ------------------------------------------------------------------
    # TransformSequenceProtocol implementation
    # ------------------------------------------------------------------

    @property
    def config(self) -> Dict[str, Any]:
        """
        Serialisable representation of the pipeline for cache key computation.

        Structure mirrors ConfigTransformManager's OmegaConf config:
            {transform_name: {transform_class: str, metadata: dict}, ...}

        The transform class name is included so that swapping the underlying
        transform implementation (same name, different class) invalidates the
        cache key.
        """
        return self._config

    def get_transforms(self) -> OrderedDict[str, DataTransform]:
        """
        Return all transforms in pipeline order.

        Returns
        -------
        OrderedDict
            Ordered mapping of transform names to `DataTransform` objects.
        """
        return self._transforms.copy()

    def get_cache_point_names(self) -> List[str]:
        """
        Return names of transforms marked as cache points.

        Returns
        -------
        list of str
            Cache boundary transform names in pipeline order.
        """
        return [
            name
            for name, dt in self._transforms.items()
            if dt.metadata.get("cache_point", False)
        ]

    def get_transform_names_after(self, transform_name: str) -> List[str]:
        """
        Return transform names that come after a named transform.

        Parameters
        ----------
        transform_name : str
            Boundary transform name.

        Returns
        -------
        list of str
            Transform names after the boundary.
        """
        names = list(self._transforms.keys())
        try:
            idx = names.index(transform_name)
            return names[idx + 1 :]
        except ValueError:
            return list(names)

    def get_config_up_to_and_including(self, transform_name: str) -> Dict:
        """
        Return a serializable config slice up to a named transform.

        Parameters
        ----------
        transform_name : str
            Last transform name to include.

        Returns
        -------
        dict
            Serializable config slice used for boundary cache keys.
        """
        names = list(self._transforms.keys())
        try:
            idx = names.index(transform_name)
        except ValueError:
            idx = len(names) - 1
        subset = {k: self._config[k] for k in names[: idx + 1]}
        return ensure_serializable(subset)

    # ------------------------------------------------------------------
    # Standalone run (no caching)
    # ------------------------------------------------------------------

    def run(
        self,
        data: SplitDatasetContainer,
    ) -> SplitDatasetContainer:
        """
        Apply all transforms sequentially.

        Use this for library usage without caching.  For caching, pass the
        TransformPipeline to PreProcessor and call PreProcessor.pipeline().

        Parameters
        ----------
        data : SplitDatasetContainer
            Input split container to transform.

        Returns
        -------
        SplitDatasetContainer
            Transformed data.
        """
        for name, transform in self._transforms.items():
            logger.info("TransformPipeline: applying '%s'.", name)
            data, _ = transform.forward(data)
        return data

    # ------------------------------------------------------------------
    # Convenience: iterate, index, length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._transforms)

    def __contains__(self, name: str) -> bool:
        return name in self._transforms

    def __repr__(self) -> str:
        names = list(self._transforms.keys())
        cache_points = self.get_cache_point_names()
        return (
            f"TransformPipeline("
            f"transforms={names}, "
            f"cache_points={cache_points})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config(
        transforms: OrderedDict[str, DataTransform],
    ) -> Dict[str, Any]:
        """
        Build a serializable config dict from `DataTransform` objects.

        Parameters
        ----------
        transforms : OrderedDict
            Ordered mapping of transform names to `DataTransform` objects.

        Returns
        -------
        dict
            Serializable pipeline config used for cache key computation.
        """
        config: Dict[str, Any] = {}
        for name, dt in transforms.items():
            config[name] = {
                "transform_class": type(dt.transform_instance).__name__,
                "metadata": ensure_serializable(dt.metadata),
            }
        return config
