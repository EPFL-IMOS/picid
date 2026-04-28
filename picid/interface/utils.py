import logging
import time
from typing import Optional, Any, Union

import hydra
import numpy as np
import torch
from joblib import Memory
from lightning import LightningDataModule
from numpy import ndarray
import awkward as ak
from omegaconf import OmegaConf, DictConfig
from torch import nn

from picid.data.data_objects import DatasetContainer, SplitViewPolicy
from picid.evaluator.base import AbstractEvaluator
from picid.exceptions import TransformError
from picid.model.adapters.base import (
    AbstractFeedForwardWrapper,
    AbstractFeedForwardTrainingWrapper,
    AbstractFitPredictWrapper,
)
from picid.pipeline.base import (
    ConstantLossLightningModule,
    TrainingLightningModule,
    FitPredictWrapperLightningModule,
)
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base.multisource import InverseTransformMixin
from picid.utils.awkward_utils import get_ak_shape
from picid.utils.rich_output import (
    descriptive_dict_differences_str,
    print_transforms_summary,
    to_descriptive_dict,
    transform_log_to_summary_string,
)

log = logging.getLogger(__name__)


def register_data_dim_resolver(
    data: dict[str, Union[np.array, ak.Array, torch.Tensor]],
) -> None:
    """Register the ``infer_data_dim`` OmegaConf resolver against a data dict.

    The resolver is used in Hydra configs to dynamically read tensor dimensions
    at composition time, e.g. ``${infer_data_dim:features,1}`` resolves to the
    size of axis 1 of ``data["features"]``. Calling this function a second time
    replaces any previously registered resolver.

    Parameters
    ----------
    data : dict[str, np.ndarray or ak.Array or torch.Tensor]
        Mapping from tensor name to array for the *training* split. The
        resolver looks up keys in this dict.
    """

    def infer_data_dim(obj: Union[np.array, ak.Array, torch.Tensor], dim: int) -> int:
        if isinstance(obj, list):
            dims = {infer_data_dim(o, dim) for o in obj}
            if len(dims) > 1:
                raise ValueError(f"Inconsistent dims found in list: {dims}")
            else:
                return dims.pop()
        elif isinstance(obj, np.ndarray):
            return obj.shape[dim]
        elif torch.is_tensor(obj):
            return tuple(obj.size())[dim]
        # elif ak.is_awkward(data):
        #     return data.shape[dim]
        elif isinstance(obj, ak.Array):
            return get_ak_shape(obj)[dim]
        else:
            raise ValueError(f"Cannot infer data dim for type {type(obj)}")

    def get_key(data, key):
        if key not in data:
            raise KeyError(
                f"Resolver issue: key: '{key}' "
                f"not found in data dictionary ({data.keys()})."
            )
        return data[key]

    OmegaConf.register_new_resolver(
        "infer_data_dim",
        lambda key, dim: infer_data_dim(get_key(data, key), dim),
        replace=True,
    )


def create_lightning_module(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    evaluators: list[AbstractEvaluator],
    loss: Any,
    backbone: nn.Module = None,
):
    """Wrap a backbone in the appropriate PyTorch Lightning module.

    Inspects the type of ``backbone`` and routes it to the correct Lightning
    wrapper. If ``backbone`` is ``None``, the model is instantiated from
    ``cfg.model`` using Hydra.

    Parameters
    ----------
    cfg : DictConfig
        Fully composed Hydra configuration (output of ``hydra.compose``).
    datamodule : LightningDataModule
        The datamodule for the current experiment (used for fit-predict models
        that need dataloader references at construction time).
    evaluators : list[AbstractEvaluator]
        Evaluator instances for each split, passed into the Lightning module.
    loss : Any
        Instantiated loss function.
    backbone : nn.Module or None
        Pre-instantiated backbone. When ``None``, Hydra instantiates the model
        from ``cfg.model``.

    Returns
    -------
    LightningModule
        A Lightning module ready to be passed to ``trainer.fit()``.

    Raises
    ------
    ValueError
        If ``backbone`` is provided but does not match any supported wrapper
        type (``AbstractFeedForwardWrapper``,
        ``AbstractFeedForwardTrainingWrapper``,
        ``AbstractFitPredictWrapper``).
    """
    # we can only partially initialize the optimizer here because
    # we need the parameter list from the model later
    # as a consequence, the same applies for the scheduler
    # because we need the optimizer instance

    optimizer_factory = hydra.utils.instantiate(
        cfg.optimization.optimizer, _partial_=True
    )

    if (
        hasattr(cfg.optimization, "scheduler")
        and cfg.optimization.scheduler is not None
    ):
        scheduler_factory = hydra.utils.instantiate(
            cfg.optimization.scheduler,
            _partial_=True,
        )
        log.info(f"Scheduler: {cfg.optimization.scheduler._target_}")

    else:
        scheduler_factory = None

    log.info(f"Optimizer: {cfg.optimization.optimizer._target_}")

    if backbone is None:
        if cfg.model.get("metadata", {}).get("lightning_model", False):
            # Model is already a standard LightningModule and does not require a wrapper
            model = hydra.utils.instantiate(
                cfg.model,
                evaluators=evaluators,
                optimizer_factory=optimizer_factory,
                scheduler_factory=scheduler_factory,
            )

            return model

        else:
            add_backbone_args = {}

            if OmegaConf.select(cfg, "datasource.data_name") == "railway":
                add_backbone_args["train_dataloader"] = datamodule.train_dataloader()
                add_backbone_args["val_dataloader"] = datamodule.val_dataloader()

            backbone = hydra.utils.instantiate(cfg.model, **add_backbone_args)

            if cfg.model.get("load_path", False):
                backbone.backbone.load_state_dict(torch.load(cfg.model.load_path))

    if isinstance(backbone, AbstractFeedForwardWrapper):
        model = ConstantLossLightningModule(
            backbone=backbone,
            loss=loss,
            evaluators=evaluators,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
        )
    elif isinstance(backbone, AbstractFeedForwardTrainingWrapper):
        model = TrainingLightningModule(
            backbone=backbone,
            loss=loss,
            evaluators=evaluators,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
        )

    elif isinstance(backbone, AbstractFitPredictWrapper):
        model = FitPredictWrapperLightningModule(
            backbone=backbone,
            loss=loss,
            evaluators=evaluators,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            debug=cfg.get("debug", False),
        )
    else:
        raise ValueError(f"Unsupported backbone wrapper type: {type(backbone)}")
    # else:
    #     model = None
    #
    #     if 'model' in cfg and not cfg.model.get("metadata", False):
    #         add_backbone_args = {}
    #
    #         if backbone is not None:
    #             backbone_wrapper = backbone
    #         else:
    #             if cfg.datasource.data_name == "railway":
    #                 add_backbone_args["train_dataloader"] = datamodule.train_dataloader()
    #                 add_backbone_args["val_dataloader"] = datamodule.val_dataloader()
    #
    #             backbone_wrapper = hydra.utils.instantiate(cfg.model, **add_backbone_args)
    #
    #             if cfg.model.get("load_path", False):
    #                 backbone_wrapper.backbone.load_state_dict(torch.load(cfg.model.load_path))
    #
    #         if isinstance(backbone_wrapper, AbstractFeedForwardWrapper):
    #             model = ConstantLossLightningModule(
    #                 backbone=backbone_wrapper,
    #                 loss=loss,
    #                 evaluators=evaluators,
    #                 optimizer_factory=optimizer_factory,
    #                 scheduler_factory=scheduler_factory,
    #             )
    #         elif isinstance(backbone_wrapper, AbstractFeedForwardTrainingWrapper):
    #             model = TrainingLightningModule(
    #                 backbone=backbone_wrapper,
    #                 loss=loss,
    #                 evaluators=evaluators,
    #                 optimizer_factory=optimizer_factory,
    #                 scheduler_factory=scheduler_factory,
    #             )
    #
    #         elif isinstance(backbone_wrapper, AbstractFitPredictWrapper):
    #             model = FitPredictWrapperLightningModule(
    #                 backbone=backbone_wrapper,
    #                 loss=loss,
    #                 evaluators=evaluators,
    #                 optimizer_factory=optimizer_factory,
    #                 scheduler_factory=scheduler_factory,
    #                 debug=cfg.get("debug", False),
    #             )
    #         else:
    #             raise ValueError(
    #                 f"Unsupported backbone wrapper type: {type(backbone_wrapper)}"
    #             )
    #
    #     else:
    #         if cfg.model.metadata.get("lightning_model", False):
    #             # Model is already a standard LightningModule and does not require a wrapper
    #             model = hydra.utils.instantiate(
    #                 cfg.model,
    #                 evaluators=evaluators,
    #                 optimizer_factory=optimizer_factory,
    #                 scheduler_factory=scheduler_factory,
    #             )

    assert model is not None, "Model could not be instantiated."
    return model


class ProcessedDatasource:
    """
    A wrapper around a DatasetContainer that holds processed data along with
    its metadata and task configuration.

    Attribute lookups that are not found on this class are transparently
    delegated to the underlying DatasetContainer instance, so callers can
    treat a ProcessedDatasource as a drop-in replacement for the original
    datasource while still accessing the processed state.

    Args:
        datasource (DatasetContainer): The original dataset container being wrapped.
        task_mode (str): The mode describing the kind of task this data is prepared for
            (e.g. "classification", "regression").
        data_dict (dict): A dictionary containing the processed data splits or arrays
            (e.g. {"train": ..., "val": ..., "test": ...}).
        meta_data_dict (dict): A dictionary containing metadata about the processed data
            (e.g. class counts, feature names, normalization statistics).

    Attributes:
        task_mode (str): Publicly accessible task mode string passed at construction.

    Example:
        >>> container = DatasetContainer(...)
        >>> processed = ProcessedDatasource(
        ...     datasource=container,
        ...     task_mode="classification",
        ...     data_dict={"train": X_train, "test": X_test},
        ...     meta_data_dict={"n_classes": 10}
        ... )
        >>> processed.data_dict["train"]      # access processed training data
        >>> processed.meta_data_dict          # access metadata
        >>> processed.some_datasource_attr    # transparently forwarded to container
    """

    def __init__(
        self,
        datasource: DatasetContainer,
        task_mode: str,
        data_dict: dict,
        meta_data_dict: dict,
    ):
        self._datasource = datasource

        self._meta_data_dict = meta_data_dict
        self._data_dict = data_dict
        self.task_mode = task_mode

    @property
    def data_dict(self):
        """dict: The processed data dictionary provided at construction."""
        return self._data_dict

    @property
    def meta_data_dict(self):
        """dict: The metadata dictionary provided at construction."""
        return self._meta_data_dict

    @property
    def datasource(self):
        """DatasetContainer: The original datasource this instance wraps."""
        return self._datasource

    def __getattr__(self, item):
        """
        Fall back to the wrapped DatasetContainer for unknown attribute lookups.

        This is called only when normal attribute resolution fails, meaning the
        attribute was not found on the ProcessedDatasource instance itself. In
        that case the lookup is forwarded to `self._datasource`.

        Args:
            item (str): Name of the attribute being looked up.

        Returns:
            The value of the attribute on the underlying datasource.

        Raises:
            AttributeError: If the attribute is not found on either this instance
                or the underlying datasource.
        """
        try:
            return getattr(self._datasource, item)
        except AttributeError:
            return getattr(self._datasource, item)


class InterfacePreProcessor:
    """
    InterfacePreProcessor is a data preprocessing class that orchestrates the loading, splitting,
    transformation, and management of datasources.

        See also:
            - `picid.data.preprocessing.single_source_loader.SingleSourceLoader`:
                    For preprocessing pipelines that operate on a single data source.
            - `picid.data.preprocessing.multi_source_loader.MultiSourceLoader`:
                    For pipelines that handle multiple data sources simultaneously.
            - `picid.data.preprocessing.phmd_loader.PHMDLoader`:
                    For specialized loading of PHMD-formatted data sources.

    Attributes:
        mode (str): Specifies the preprocessing mode, either 'per_unit' or 'cross_unit'.
        datasource: The data source object providing access to raw data and metadata.
        transforms: A collection of DataTransform objects to be applied sequentially.
        data (Optional[DatasetContainer]): The processed dataset container.

    Methods:
        __init__(mode, datasource, transforms, **kwargs):
            Initializes the PreProcessor with the specified mode, datasource, and transforms.
        get_meta_data_dict():
            Retrieves metadata from the datasource as a dictionary.
        get_processed_data_dict(return_splits_on_first_level: bool = False) -> dict:
            Returns the processed data as a dictionary, optionally splitting data at the first level.
        fetch_data() -> None:
            Loads and stores data and metadata from the datasource.
        apply_transforms(data: DatasetContainer, transforms) -> List[Any]:
            Applies a sequence of data transformations to the dataset.
        pipeline() -> DatasetContainer:
            Executes the full preprocessing pipeline: loads, splits, fetches,
            transforms, and returns the processed data.
    """

    def __init__(self, datasource, transforms, cache_path=None, **kwargs):
        self.datasource = datasource
        self.transforms = transforms
        self.memory = Memory(cache_path, verbose=20)

        self._is_preprocessed = False
        self.data: Optional[DatasetContainer] = None
        self.meta_data: dict = {}

    def get_meta_data_dict(self):
        return self.meta_data

    def get_processed_data_dict(
        self, return_splits_on_first_level: bool = False
    ) -> dict[str, ndarray | list[ndarray]]:
        """
        Returns the final processed data.

        The returned dictionary is structured as follows:
        {
            "features": {
                "train": ndarray | list[ndarray],
                "val": ndarray | list[ndarray],
                "test": ndarray | list[ndarray]
            },
            "target": {
                "train": ndarray | list[ndarray],
                "val": ndarray | list[ndarray],
                "test": ndarray | list[ndarray]
            },
            "other_named_tensor": {
                ...
            }
        }

        If return_splits_on_first_level is True, the function will return the splits at the first level.
        """

        if not self._is_preprocessed or self.data is None:
            raise RuntimeError(
                "Data has not been preprocessed. Please run the pipeline() first."
            )

        if return_splits_on_first_level:
            return self.data.to_split_dict()
        else:
            return self.data

    def fetch_data(self) -> None:
        """Prepare data from the datasource and store as instance variables."""

        data: DatasetContainer | list[DatasetContainer] = self.datasource.get_data()

        assert isinstance(data, DatasetContainer), (
            f"Expected MultiDataChunk, but got {type(data)}, "
            "make sure 'datasource.get_data()' returns MultiDataChunk"
        )

        self.data: DatasetContainer = data

    @staticmethod
    def apply_transforms(
        data: DatasetContainer, transforms, caching_values: Any = None
    ) -> DatasetContainer:
        """Apply a list of transforms sequentially to a DatasetContainer.

        Parameters
        ----------
        data : DatasetContainer
            The dataset to transform.
        transforms : list[DataTransform] or None
            Ordered list of transforms to apply. Each transform is applied in
            turn and receives the output of the previous one. If ``None``,
            ``data`` is returned unchanged.
        caching_values : Any, optional
            Reserved for joblib caching; pass ``None``.

        Returns
        -------
        DatasetContainer
            The transformed dataset container.

        Raises
        ------
        ValueError
            If any element of ``transforms`` is not a ``DataTransform`` instance.
        """
        if transforms is None:
            return data

        if not all(isinstance(t, DataTransform) for t in transforms):
            raise ValueError(
                "Every element of transforms must be a DataTransform object."
            )

        summary = []

        try:
            for transform in transforms:
                log.info(
                    "Applying transform: '%s' (Strategy: %s)",
                    transform.transform_name,
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
                        f"Transform {transform.transform_name!r} failed. "
                        f"Original error: {e}",
                        step_id=transform.transform_name,
                        cause=e,
                    ) from e

                elapsed = time.time() - start_time
                log.info(
                    "Transform '%s' completed in %.4f seconds.",
                    transform.transform_name,
                    elapsed,
                )

                post_descr = to_descriptive_dict(
                    data.to_split_dict(SplitViewPolicy.KEEP_UNIT_LISTS)["train"],
                    calculate_stat=False,
                )
                log_str = transform_log_to_summary_string(transform_log)

                summary.append(
                    {
                        "transform_name": transform.transform_name,
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

    def pre_process_data(self):
        """Load the datasource and split it into train/val/test.

        Calls ``load_data()`` and ``split_data()`` on the datasource, then
        fetches the resulting ``DatasetContainer`` into ``self.data``.

        Returns
        -------
        dict
            Metadata dict returned by ``datasource.get_meta_data()``.
        """
        self.datasource.load_data()
        self.datasource.split_data()

        meta_data = self.datasource.get_meta_data()

        self.fetch_data()

        return meta_data

    def get_meta_data(self, datasource=None):
        return datasource.get_meta_data()

    def pipeline(self) -> ProcessedDatasource:
        """
        Run the preprocessing pipeline, loading the preprocessed data if already saved in the cache.
        """

        self.meta_data = self.pre_process_data()

        apply_transforms = self.memory.cache(self.apply_transforms)
        self.data = apply_transforms(self.data, self.transforms, None)

        self._is_preprocessed = True

        # self.data.meta_data_dict = self.get_meta_data_dict()
        # self.data.data_dict = self.get_processed_data_dict()

        return ProcessedDatasource(
            self.datasource,
            task_mode=self.datasource.task_mode,
            data_dict=self.data.to_split_dict(),
            meta_data_dict=self.get_meta_data_dict(),
        )


def get_inverter_for_key_with_name(
    transforms: list[DataTransform], key: str, which: str = "last"
) -> tuple[InverseTransformMixin | None, str | None]:
    """Find the first or last invertible transform that handles a given key.

    Scans ``transforms`` for ``DataTransform`` instances whose underlying
    transform implements ``InverseTransformMixin`` and whose ``assign_to``
    includes ``key``.

    Parameters
    ----------
    transforms : list[DataTransform]
        Ordered list of transforms as passed to ``process_datasource()``.
    key : str
        The data-dict key to look for (e.g. ``"rul"``).
    which : {"first", "last"}
        Whether to return the first or last matching transform.
        Default ``"last"``.

    Returns
    -------
    tuple[InverseTransformMixin or None, str or None]
        The matching transform instance and its ``transform_name``, or
        ``(None, None)`` if no match is found.

    Raises
    ------
    ValueError
        If ``which`` is not ``"first"`` or ``"last"``.
    """

    if which not in ("first", "last"):
        raise ValueError(
            f"get_inverter_for_key_with_name(which=...) must be 'first' or 'last', got {which!r}"
        )

    inverters = [
        t for t in transforms if isinstance(t.transform_instance, InverseTransformMixin)
    ]

    if len(inverters) == 0:
        return None, None

    inverter = None
    inverter_name = None
    for dt in inverters:
        assign_to = getattr(dt, "assign_to", None)
        if assign_to is None:
            continue
        keys = [assign_to] if isinstance(assign_to, str) else list(assign_to)
        if key not in keys:
            continue
        if isinstance(dt.transform_instance, InverseTransformMixin):
            inverter = dt.transform_instance
            inverter_name = dt.transform_name
            if which == "first":
                return inverter, inverter_name
    return inverter, inverter_name
