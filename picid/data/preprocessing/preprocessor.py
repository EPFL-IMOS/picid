"""
preprocessor.py
===============

PreProcessor orchestrates loading, splitting, transformation, and caching of
datasets for both framework and library users.

Framework users (run.py)
------------------------
    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    preprocessor = PreProcessor(datasource=datasource, transforms=transforms_manager)
    data = preprocessor.pipeline(cache_paths=..., cache_preprocessed=True)

Library users
-------------
    from picid.transforms.base.transform_pipeline import TransformPipeline

    pipeline = TransformPipeline([
        DataTransform("normalize", scaler,  {"apply_to": "features", "fit_on": "train"}),
        DataTransform("window",    windower, {"apply_to": "features", "cache_point": True}),
        DataTransform("scale_y",   scaler2, {"apply_to": "target",   "fit_on": "train"}),
    ])

    # Without caching:
    data = pipeline.run(data)

    # With the same three-tier caching as the framework:
    preprocessor = PreProcessor(datasource=datasource, transforms=pipeline)
    data = preprocessor.pipeline(
        data_cache_path=".cache",
        data_library_part_path="./picid/data/datasources",
        transform_library_part_path="./picid/transforms",
        cache_preprocessed=True,
    )

Both ConfigTransformManager and TransformPipeline satisfy TransformSequenceProtocol.
PreProcessor and the pipeline runners depend only on that protocol.

Pipeline runners
----------------
    DirectPipelineRunner     — no caching; load → split → fetch → transform → return.
    CachingPipelineRunner    — three-tier cache (loaded_and_splitted_data → boundary →
                               preprocessed) with optional boundary restore.

Behaviour is identical to the previous implementation; the logic is decomposed
into named methods for testability and the concrete transform manager type is
no longer hardcoded.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Union


from picid.data.cache.offline import FileSystemCache
from picid.data.data_objects import DatasetContainer, SplitViewPolicy
from picid.data.data_objects.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestEntry,
    MetadataManifest,
)
from picid.data.datasources.base.exceptions import (
    DatasourceContractError,
    DatasourceError,
)
from picid.data.datasources.base.contracts import DatasourceProtocol
from picid.data.preprocessing.base import PreProcessorInterface
from picid.exceptions import (
    TransformError,
    build_preprocessing_datasource_error,
)
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base.pipeline import _get_producer_version
from picid.transforms.base.transform_manager import ConfigTransformManager
from picid.transforms.base.transform_pipeline import (
    TransformPipeline,
    TransformSequenceProtocol,
)
from picid.utils import print_data_dict_structure
from picid.data.preprocessing.pipeline_runners import (
    CachingPipelineRunner,
    DirectPipelineRunner,
)
from picid.utils.rich_output import (
    descriptive_dict_differences_str,
    print_transforms_summary,
    to_descriptive_dict,
    transform_log_to_summary_string,
)

logger = logging.getLogger(__name__)

# Union type accepted wherever a transform sequence is expected.
AnyTransformSequence = Union[ConfigTransformManager, TransformPipeline]


# ---------------------------------------------------------------------------
# PreProcessor
# ---------------------------------------------------------------------------


class PreProcessor(PreProcessorInterface):
    """
    Orchestrate loading, splitting, transformation, and caching.

    The class accepts either a Hydra-driven transform manager or a pure Python
    transform pipeline. Both satisfy the same transform-sequence protocol, so
    the runner logic can stay agnostic to the concrete implementation.

    Parameters
    ----------
    datasource : DatasourceProtocol
        Datasource object that materializes the raw and split dataset.
    transforms : ConfigTransformManager | TransformPipeline | None, optional
        Transform sequence to apply. When omitted, an empty transform manager
        is created so callers can still run the pipeline.
    **kwargs : Any
        Forwarded for compatibility with the interface base class.

    Attributes
    ----------
    datasource : DatasourceProtocol
        Datasource object that materializes the raw and split dataset.
    transforms : ConfigTransformManager | TransformPipeline
        Ordered transform sequence applied to the loaded dataset.
    data : DatasetContainer | None
        Final processed dataset container after ``pipeline()`` completes.

    Notes
    -----
    The legacy ``mode`` parameter (``"per_unit"`` / ``"cross_unit"``) has
    been removed. If split mode is needed for logging or evaluator
    configuration, read it from the datasource before constructing the
    preprocessor, for example ``split_mode = datasource.get_split_mode()``.
    """

    def __init__(
        self,
        datasource: DatasourceProtocol,
        transforms: Optional[AnyTransformSequence] = None,
        **kwargs,
    ):
        if transforms is None:
            transforms = ConfigTransformManager(transforms_config=None)
            logger.info("No transforms provided; using empty ConfigTransformManager.")
        elif not isinstance(transforms, TransformSequenceProtocol):
            raise TypeError(
                f"transforms must be a ConfigTransformManager or TransformPipeline "
                f"(or any object satisfying TransformSequenceProtocol), "
                f"got {type(transforms).__name__}."
            )

        super().__init__(datasource=datasource, transforms=transforms)

        self._is_preprocessed = False
        self.data: Optional[DatasetContainer] = None
        self.meta_data: dict = {}
        self.cache = FileSystemCache()

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_meta_data_dict(self) -> dict:
        return self.meta_data

    def get_processed_data_container(self) -> DatasetContainer:
        """
        Return the final processed dataset container.

        Returns
        -------
        DatasetContainer
            Fully processed dataset container produced by ``pipeline()``.
        """
        if not self._is_preprocessed or self.data is None:
            raise RuntimeError("Data has not been preprocessed. Run pipeline() first.")
        return self.data

    def get_processed_split_dict(
        self,
        view_policy: SplitViewPolicy = SplitViewPolicy.KEEP_UNIT_LISTS,
    ) -> Dict[str, Any]:
        """
        Return the processed data as a split-first dictionary view.

        Parameters
        ----------
        view_policy : SplitViewPolicy, default=SplitViewPolicy.KEEP_UNIT_LISTS
            Controls how unit lists are exposed in the returned split view.

        Returns
        -------
        dict[str, Any]
            Split-first representation of the processed dataset.
        """
        container = self.get_processed_data_container()
        container.validate()
        return container.to_split_dict(view_policy=view_policy)

    def get_cached_transform_manager(self) -> AnyTransformSequence:
        """
        Return the current transform sequence, cached or original.

        Returns
        -------
        ConfigTransformManager | TransformPipeline
            Transform sequence used by the preprocessor.
        """
        return self.transforms

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def fetch_data(self) -> None:
        """
        Load data from the datasource and attach a manifest entry.

        Raises
        ------
        PreprocessingDatasourceError
            If the datasource raises a typed datasource exception while
            materializing the dataset container.
        """
        data = self._run_datasource_step("get_data", self.datasource.get_data)

        if not isinstance(data, DatasetContainer):
            contract_error = DatasourceContractError(
                f"datasource.get_data() must return a DatasetContainer, "
                f"got {type(data).__name__}."
            )
            raise build_preprocessing_datasource_error(
                "get_data", self.datasource, contract_error
            ) from contract_error

        self.data = data
        self._add_datasource_manifest_entry()

    def _run_datasource_step(
        self,
        stage: str,
        operation: Callable[[], Any],
    ) -> Any:
        """
        Execute one datasource operation with preprocessing-stage context.

        Parameters
        ----------
        stage : str
            Datasource operation being executed, for example ``"load_data"``.
        operation : Callable[[], Any]
            Zero-argument callable that performs the datasource action.

        Returns
        -------
        Any
            Result produced by the datasource operation.

        Raises
        ------
        PreprocessingDatasourceError
            If the datasource raises a typed datasource exception.

        Notes
        -----
        Assertion-based invariant failures are intentionally not caught here so
        programming bugs and data-corruption assertions keep their original
        signal.
        """

        try:
            return operation()
        except DatasourceError as exc:
            raise build_preprocessing_datasource_error(
                stage, self.datasource, exc
            ) from exc

    def _add_datasource_manifest_entry(self) -> None:
        """
        Append a datasource provenance entry to ``self.data.manifest``.

        The manifest entry is only added when the processed container exposes a
        metadata manifest. This keeps the provenance behavior a no-op for
        containers that do not use the manifest subsystem.
        """
        if self.data is None:
            return
        manifest = getattr(self.data, "manifest", None)
        if not isinstance(manifest, MetadataManifest):
            return

        ds = self.datasource
        payload: Dict[str, Any] = {"datasource_type": type(ds).__name__}

        data_name = self._resolve_datasource_name(ds)
        if data_name is not None:
            payload["data_name"] = data_name
        if hasattr(ds, "task_mode"):
            payload["task_mode"] = getattr(ds, "task_mode", None)

        entry = ManifestEntry(
            schema_version=MANIFEST_SCHEMA_VERSION,
            producer_version=_get_producer_version(),
            category="datasource",
            payload=payload,
            step_id=type(ds).__name__,
            key=None,
            split=None,
        )
        manifest.add(entry)

    @staticmethod
    def _resolve_datasource_name(ds: Any) -> Optional[Any]:
        """
        Return the datasource name exposed by the most specific accessor.

        Parameters
        ----------
        ds : Any
            Datasource object to inspect.

        Returns
        -------
        str | list[str] | None
            Name reported by ``data_name``, ``get_data_names()``, or
            ``get_data_name()``, preferring the most explicit accessor.
        """
        if hasattr(ds, "data_name"):
            val = getattr(ds, "data_name", None)
            if isinstance(val, str):
                return val

        if hasattr(ds, "get_data_names") and callable(getattr(ds, "get_data_names")):
            try:
                val = ds.get_data_names()
                if isinstance(val, tuple) and len(val) == 1:
                    return val[0]
                if isinstance(val, tuple) and all(isinstance(x, str) for x in val):
                    return list(val)
            except Exception:
                pass

        if hasattr(ds, "get_data_name") and callable(getattr(ds, "get_data_name")):
            try:
                val = ds.get_data_name()
                if isinstance(val, str):
                    return val
                if isinstance(val, (list, tuple)) and all(
                    isinstance(x, str) for x in val
                ):
                    return val
            except Exception:
                pass

        return None

    # ------------------------------------------------------------------
    # Transform application
    # ------------------------------------------------------------------

    def apply_transforms(
        self,
        data: DatasetContainer,
        transforms: Dict[str, DataTransform],
        after_each_transform_callback: Optional[
            Callable[[DatasetContainer, str], None]
        ] = None,
    ) -> DatasetContainer:
        """
        Apply an ordered transform mapping to a dataset container.

        Parameters
        ----------
        data : DatasetContainer
            Container to transform in place.
        transforms : OrderedDict[str, DataTransform]
            Already-resolved transforms returned by ``get_transforms()``.
        after_each_transform_callback : callable, optional
            Optional callback invoked after each successful transform.

        Returns
        -------
        DatasetContainer
            Transformed dataset container.

        Raises
        ------
        ValueError
            If any value in transforms is not a DataTransform.
        TransformError
            If a transform raises during application.
        """
        if not all(isinstance(t, DataTransform) for t in transforms.values()):
            raise ValueError(
                "Every element of transforms must be a DataTransform instance."
            )

        summary = []

        try:
            for transform_name, transform in transforms.items():
                logger.info(
                    "Applying transform: '%s' (Strategy: %s)",
                    transform_name,
                    transform.strategy.__class__.__name__,
                )

                pre_descr = to_descriptive_dict(
                    data.to_split_dict(SplitViewPolicy.KEEP_UNIT_LISTS)["train"],
                    calculate_stat=False,
                )
                start_time = time.time()

                try:
                    data, transform_log = transform.forward(data)
                except TransformError:
                    raise
                except Exception as e:
                    raise TransformError(
                        f"Transform {transform_name!r} failed. " f"Original error: {e}",
                        step_id=transform_name,
                        cause=e,
                    ) from e

                if after_each_transform_callback is not None:
                    after_each_transform_callback(data, transform_name)

                elapsed = time.time() - start_time
                logger.info(
                    "Transform '%s' completed in %.4f seconds.",
                    transform.transform_name,
                    elapsed,
                )

                post_descr = to_descriptive_dict(
                    data.to_split_dict(SplitViewPolicy.KEEP_UNIT_LISTS)["train"],
                    calculate_stat=False,
                )
                log_str = self._extract_log_string(transform_log)

                summary.append(
                    {
                        "transform_name": transform_name,
                        "time": f"{elapsed:.4f}",
                        "status": "Success",
                        "details": log_str,
                        "changes": str(
                            descriptive_dict_differences_str(
                                pre_descr, post_descr, mode="changed"
                            )
                        ),
                        "added": str(
                            descriptive_dict_differences_str(
                                pre_descr, post_descr, mode="added"
                            )
                        ),
                        "removed": str(
                            descriptive_dict_differences_str(
                                pre_descr, post_descr, mode="removed"
                            )
                        ),
                        "inputs": str(pre_descr),
                    }
                )
        finally:
            print_transforms_summary(summary)

        return data

    @staticmethod
    def _extract_log_string(transform_log: Any) -> str:
        """
        Extract a human-readable summary from a transform log payload.

        Parameters
        ----------
        transform_log : Any
            Log payload returned by a transform.

        Returns
        -------
        str
            Human-readable log summary.
        """
        return transform_log_to_summary_string(transform_log)

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def pipeline(
        self,
        data_cache_path: str = None,
        data_library_part_path: str = None,
        transform_library_part_path: str = None,
        cache_preprocessed: bool = False,
        save_boundary_caches: bool = True,
    ) -> DatasetContainer:
        """
        Run the full preprocessing pipeline.

        When all cache paths and cache_preprocessed=True are provided, a
        CachingPipelineRunner is used.  Otherwise a DirectPipelineRunner is used.
        Both runners work with ConfigTransformManager and TransformPipeline.

        Parameters
        ----------
        data_cache_path : str, optional
            Root directory for all cache files.
        data_library_part_path : str, optional
            Path to datasource library (used for cache key computation).
        transform_library_part_path : str, optional
            Path to transform library (used for cache key computation).
        cache_preprocessed : bool
            Enable the preprocessed (final) cache.
        save_boundary_caches : bool
            When True (default), save a boundary cache after each cache-point
            transform so future runs can restore from there.

        Returns
        -------
        DatasetContainer
            Final processed dataset container.

        Raises
        ------
        PreprocessingDatasourceError
            If a datasource-stage operation fails with a typed datasource
            exception while preprocessing is materializing the dataset.
        """
        cache_enabled = bool(
            data_cache_path
            and transform_library_part_path
            and data_library_part_path
            and cache_preprocessed
        )

        if cache_enabled:
            runner: Union[DirectPipelineRunner, CachingPipelineRunner] = (
                CachingPipelineRunner(
                    preprocessor=self,
                    data_cache_path=data_cache_path,
                    data_library_part_path=data_library_part_path,
                    transform_library_part_path=transform_library_part_path,
                    save_boundary_caches=save_boundary_caches,
                )
            )
        else:
            runner = DirectPipelineRunner(self)

        self.data = runner.run()
        self._is_preprocessed = True

        logger.debug(
            "Data structure after transforms:\n%s",
            print_data_dict_structure(self.data),
        )
        return self.data
