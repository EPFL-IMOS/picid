from typing import Optional, Any, List, Union

import hydra
import numpy as np
import torch
from joblib import Memory
from lightning import LightningDataModule
from numpy import ndarray
import awkward as ak
from omegaconf import OmegaConf, DictConfig
from torch import nn

from picid.data.data_objects.data import DatasetContainer
from picid.evaluator.base import AbstractEvaluator
from picid.interface.model.custom_model import CustomModelTrainer
from picid.model.wrappers.base import AbstractFeedForwardWrapper, AbstractFeedForwardTrainingWrapper, \
    AbstractFitPredictWrapper
from picid.pipeline.base import ConstantLossLightningModule, TrainingLightningModule, FitPredictWrapperLightningModule
from picid.transforms.base.data_transform import DataTransform
from picid.utils.awkward_utils import get_ak_shape


def register_data_dim_resolver(
    data: dict[str, Union[np.array, ak.Array, torch.Tensor]],
) -> None:

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
    backbone: nn.Module =None
):
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
    else:
        scheduler_factory = None

    if isinstance(backbone, CustomModelTrainer):

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
            raise ValueError(
                f"Unsupported backbone wrapper type: {type(backbone)}"
            )
    else:
        model = None

        if 'model' in cfg and not cfg.model.get("metadata", False):
            add_backbone_args = {}

            if backbone is not None:
                backbone_wrapper = backbone
            else:
                if cfg.datasource.data_name == "railway":
                    add_backbone_args["train_dataloader"] = datamodule.train_dataloader()
                    add_backbone_args["val_dataloader"] = datamodule.val_dataloader()

                backbone_wrapper = hydra.utils.instantiate(cfg.model, **add_backbone_args)

                if cfg.model.get("load_path", False):
                    backbone_wrapper.backbone.load_state_dict(torch.load(cfg.model.load_path))

            if isinstance(backbone_wrapper, AbstractFeedForwardWrapper):
                model = ConstantLossLightningModule(
                    backbone=backbone_wrapper,
                    loss=loss,
                    evaluators=evaluators,
                    optimizer_factory=optimizer_factory,
                    scheduler_factory=scheduler_factory,
                )
            elif isinstance(backbone_wrapper, AbstractFeedForwardTrainingWrapper):
                model = TrainingLightningModule(
                    backbone=backbone_wrapper,
                    loss=loss,
                    evaluators=evaluators,
                    optimizer_factory=optimizer_factory,
                    scheduler_factory=scheduler_factory,
                )

            elif isinstance(backbone_wrapper, AbstractFitPredictWrapper):
                model = FitPredictWrapperLightningModule(
                    backbone=backbone_wrapper,
                    loss=loss,
                    evaluators=evaluators,
                    optimizer_factory=optimizer_factory,
                    scheduler_factory=scheduler_factory,
                    debug=cfg.get("debug", False),
                )
            else:
                raise ValueError(
                    f"Unsupported backbone wrapper type: {type(backbone_wrapper)}"
                )

        else:
            if cfg.model.metadata.get("lightning_model", False):
                # Model is already a standard LightningModule and does not require a wrapper
                model = hydra.utils.instantiate(
                    cfg.model,
                    evaluators=evaluators,
                    optimizer_factory=optimizer_factory,
                    scheduler_factory=scheduler_factory,
                )

    assert model is not None, "Model could not be instantiated."
    return model

def create_lightning_module_from_backbone(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    evaluators: list[AbstractEvaluator],
    loss: Any,
):
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
    else:
        scheduler_factory = None

    model = None
    if not cfg.model.get("metadata", False):
        add_backbone_args = {}
        if cfg.datasource.data_name == "railway":
            add_backbone_args["train_dataloader"] = datamodule.train_dataloader()
            add_backbone_args["val_dataloader"] = datamodule.val_dataloader()

        backbone_wrapper = hydra.utils.instantiate(cfg.model, **add_backbone_args)

        if cfg.model.get("load_path", False):
            backbone_wrapper.backbone.load_state_dict(torch.load(cfg.model.load_path))

        if isinstance(backbone_wrapper, AbstractFeedForwardWrapper):
            model = ConstantLossLightningModule(
                backbone=backbone_wrapper,
                loss=loss,
                evaluators=evaluators,
                optimizer_factory=optimizer_factory,
                scheduler_factory=scheduler_factory,
            )
        elif isinstance(backbone_wrapper, AbstractFeedForwardTrainingWrapper):
            model = TrainingLightningModule(
                backbone=backbone_wrapper,
                loss=loss,
                evaluators=evaluators,
                optimizer_factory=optimizer_factory,
                scheduler_factory=scheduler_factory,
            )

        elif isinstance(backbone_wrapper, AbstractFitPredictWrapper):
            model = FitPredictWrapperLightningModule(
                backbone=backbone_wrapper,
                loss=loss,
                evaluators=evaluators,
                optimizer_factory=optimizer_factory,
                scheduler_factory=scheduler_factory,
                debug=cfg.get("debug", False),
            )
        else:
            raise ValueError(
                f"Unsupported backbone wrapper type: {type(backbone_wrapper)}"
            )

    else:
        if cfg.model.metadata.get("lightning_model", False):
            # Model is already a standard LightningModule and does not require a wrapper
            model = hydra.utils.instantiate(
                cfg.model,
                evaluators=evaluators,
                optimizer_factory=optimizer_factory,
                scheduler_factory=scheduler_factory,
            )

    assert model is not None, "Model could not be instantiated."
    return model

class ProcessedDatasource:
    def __init__(self,
                 datasource: DatasetContainer,
                 task_mode: str,
                 data_dict: dict,
                 meta_data_dict: dict):

        self._datasource = datasource

        self._meta_data_dict = meta_data_dict
        self._data_dict = data_dict
        self.task_mode = task_mode

    @property
    def data_dict(self):
        return self._data_dict

    @property
    def meta_data_dict(self):
        return self._meta_data_dict

    @property
    def datasource(self):
        return self._datasource

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self._datasource, item)


class InterfacePreProcessor:
    """
        InterfacePreProcessor is a data preprocessing class that orchestrates the loading, splitting,
        transformation, and management of datasources.

        See also:
            - `lmetk.data.preprocessing.single_source_loader.SingleSourceLoader`:
                    For preprocessing pipelines that operate on a single data source.
            - `lmetk.data.preprocessing.multi_source_loader.MultiSourceLoader`:
                    For pipelines that handle multiple data sources simultaneously.
            - `lmetk.data.preprocessing.phmd_loader.PHMDLoader`:
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
    def     __init__(self, datasource, transforms, cache_path=None, **kwargs):

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
    def apply_transforms(data: DatasetContainer, transforms, caching_values: Any = None) -> List[Any]:
        """
        Apply a list of transforms to each data item in data.
        Args:
            data: List of data items.
            transforms_lst: List of lists of DataTransform objects, one list per data item.
        Returns:
            List of transformed data items.
        """
        if transforms is None:
            return data

        if not all(isinstance(t, DataTransform) for t in transforms):
            raise ValueError(
                "Every element of transforms_lst must be a DataTransform object."
            )

        try:
            for transform in transforms:
                data, transform_log = transform.forward(data)
        except Exception as e:
            print(e)
        return data

    def pre_process_data(self):
        self.datasource.load_data()
        self.datasource.split_data()

        meta_data = self.datasource.get_meta_data()

        self.fetch_data()

        return meta_data

    def get_meta_data(self, datasource=None):

        return datasource.get_meta_data()

    def pipeline(
        self
    ) -> ProcessedDatasource:
        """
        Run the preprocessing pipeline, loading the preprocessed data if already saved in the cache.
        """

        self.meta_data = self.pre_process_data()

        apply_transforms = self.memory.cache(self.apply_transforms)
        self.data = apply_transforms(self.data, self.transforms, None)

        self._is_preprocessed = True

        # self.data.meta_data_dict = self.get_meta_data_dict()
        # self.data.data_dict = self.get_processed_data_dict()

        return ProcessedDatasource(self.datasource,
                                   task_mode=self.datasource.task_mode,
                                   data_dict=self.data.to_split_dict(),
                                   meta_data_dict=self.get_meta_data_dict())
