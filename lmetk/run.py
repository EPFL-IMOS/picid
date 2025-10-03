"""Main entry point for training and testing models."""

import os
import random
from pathlib import Path

# shutil
from typing import Any, Tuple, Union
import uuid

import hydra
import lightning as L
import numpy as np
import torch
import awkward as ak

from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from hydra.core.hydra_config import HydraConfig

from lmetk.data.datasources.base.multi_source_loader import MultiSourceLoader
from lmetk.data.datasources.base.phmd_loader import PHMDMultiSourceLoader
from lmetk.data.datasources.base.single_source_loader import SingleSourceLoader
from lmetk.data.datasources.kubric_movie.loader import MOViLoader
from lmetk.data.datasources.nb14.loader import NB14Loader
from lmetk.data.datasources.unibo.loader import UNIBO21Loader

from lmetk.data.preprocessing.preprocessor import PreProcessor
from lmetk.model.wrappers.base import (
    AbstractFeedForwardWrapper,
    AbstractFitPredictWrapper,
    AbstractFeedForwardTrainingWrapper,
)
from lmetk.pipeline.base import (
    ConstantLossLightningModule,
    FitPredictWrapperLightningModule,
    TrainingLightningModule,
)
from lmetk.callbacks.model_checkpoint import ModelCheckpointWithConfig
from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.transforms.base.transform_manager import ConfigTransformManager


from lmetk.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
    log_hyperparameters,
    display_targets,
)
from lmetk.utils.logging_utils import get_hydra_override
from lmetk.utils.awkward_utils import get_ak_shape
from lmetk.utils.examine_model import get_model_summary
from lmetk.model.autoencoder.utils import get_autoencoder_summary_and_tags

# register resolver before hydra.main
OmegaConf.register_new_resolver("flat", lambda s: s.replace("/", "+"))
OmegaConf.register_new_resolver("uuid", lambda kind="short": _uuid(kind))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))


OmegaConf.register_new_resolver("prod", lambda *args: np.prod(args))


def safe_div(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    if a % b != 0:
        raise ValueError(f"{a} is not divisible by {b}")
    return a // b


OmegaConf.register_new_resolver("quot", lambda a, b: safe_div(a, b))


# safe diff that returns a if a or b are strings
def diff_if_not_string(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a
    return a - b


OmegaConf.register_new_resolver("diff", lambda a, b: diff_if_not_string(a, b))


OmegaConf.register_new_resolver(
    "infer_data_dim", lambda key, dim: "not yet initialized"
)


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


def _uuid(kind: str) -> str:
    u = uuid.uuid4()
    if kind == "hex":
        return u.hex
    if kind == "short":
        return u.hex[:8]
    raise ValueError(f"Unknown uuid kind: {kind}")


def initialize_hydra() -> DictConfig:
    """Initialize Hydra when main is not an option (e.g. tests).

    Returns
    -------
    DictConfig
        A DictConfig object containing the config tree.
    """
    hydra.initialize(version_base="1.3", config_path="../configs", job_name="run")
    cfg = hydra.compose(config_name="run.yaml")
    return cfg


torch.set_num_threads(1)
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the model.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("num_threads", None) is not None:
        torch.set_num_threads(cfg.num_threads)
        os.environ["OMP_NUM_THREADS"] = str(cfg.num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cfg.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(cfg.num_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cfg.num_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cfg.num_threads)
        log.info(f"Set number of threads to {cfg.num_threads}")

    # Set seed for random number generators in pytorch, numpy and python.random
    L.seed_everything(cfg.seed, workers=True)
    # Seed for torch
    torch.manual_seed(cfg.seed)
    # Seed for numpy
    np.random.seed(cfg.seed)
    # Seed for python random
    random.seed(cfg.seed)

    for k, d in cfg.paths.items():
        path = Path(d).expanduser().resolve()
        if not path.exists():
            log.warning(f"Directory {k} created at: {path.as_uri()}")
            os.makedirs(path, exist_ok=True)
        log.info(f"Directory for {k} is: {path.as_uri()}")

    # HydraConfig.set_config(cfg)
    # display_config_sources(cfg)
    display_targets(cfg)
    if cfg.model._target_ == "lmetk.model.autoencoder.wrapper.AutoencoderWrapper":
        model_summary_table, model_tags_list = get_autoencoder_summary_and_tags(
            cfg.model
        )
        cfg.logger.wandb["tags"] = [tag for tag in model_tags_list]
        log.info(model_summary_table)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiate and load dataset
    log.info(f"Instantiating loader <{cfg.datasource._target_}>")
    dataset_source: SingleSourceLoader | MultiSourceLoader = hydra.utils.instantiate(
        cfg.datasource
    )
    # Create manager
    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    # Get transform names
    log.info("Transform names:" + str(transforms_manager.get_transform_names()))

    # Get transforms that apply to features
    # feature_transforms = transforms_manager.get_transforms_by_apply_to("features")
    # log.info("Feature transforms:" + str(list(feature_transforms.keys())))

    single_source_types = (
        SingleSourceLoader,
        PHMDMultiSourceLoader,
        MOViLoader,
        NB14Loader,
        UNIBO21Loader,
    )
    if isinstance(dataset_source, single_source_types):
        preprocessor_mode = "per_unit"

    elif isinstance(dataset_source, MultiSourceLoader):
        preprocessor_mode = "cross_unit"

    else:
        raise ValueError("Invalid dataset source type")

    preprocessor = PreProcessor(preprocessor_mode, dataset_source, transforms_manager)

    # Get cache dirs and flags
    data_cache_path = (
        cfg.paths.cache_path
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    data_library_part_path = (
        Path(cfg.paths.root_dir) / "lmetk/data/datasources"
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    transform_library_part_path = (
        Path(cfg.paths.root_dir) / "lmetk/transforms"
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    cache_preprocessed_flag = cfg.cache.use_cache_after_transfroms

    # Run the preprocessing pipeline
    preprocessor.pipeline(
        data_cache_path=data_cache_path,
        data_library_part_path=data_library_part_path,
        transform_library_part_path=transform_library_part_path,
        cache_preprocessed=cache_preprocessed_flag,
    )

    # TODO: return_splits_on_first_level is not clea..
    data_dict = preprocessor.get_processed_data_dict(return_splits_on_first_level=True)
    meta_data_dict = preprocessor.get_meta_data_dict()
    if cache_preprocessed_flag:
        transforms_manager = preprocessor.get_chached_transform_manager()

    assert set(data_dict.keys()) == {"train", "val", "test"}

    features_set = set(data_dict["train"].keys())
    for split in ["val", "test"]:
        assert set(data_dict[split].keys()) == features_set, (
            f"Mismatch in features for split {split}. "
            f"Expected: {features_set}, "
            f"Found: {set(data_dict[split].keys())}"
        )
    input_data_task_keys = cfg.task_definition.model.data_requirements.input_tensors

    # 1. Create the datasets and store them in a dictionary
    datasets = {}
    for split in ["train", "val", "test"]:
        key = f"dataset_{split}"
        datasets[key] = hydra.utils.instantiate(
            cfg.dataset,
            data_dict={key: data_dict[split][key] for key in input_data_task_keys},
            meta_data_dict=meta_data_dict,
        )

    # 2. Unpack the dictionary into the variables your code expects
    dataset_train = datasets["dataset_train"]
    dataset_val = datasets["dataset_val"]
    dataset_test = datasets["dataset_test"]

    # 3. Now, the rest of your original code works perfectly
    log.info(f"Train dataset size: {len(dataset_train)}")
    log.info(f"Validation dataset size: {len(dataset_val)}")
    log.info(f"Test dataset size: {len(dataset_test)}")

    # Instantiate dataloader
    datamodule = hydra.utils.instantiate(
        cfg.datamodule,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
    )

    # Log the sizes of one batch for each data loader
    for split, loader in zip(
        ["train", "val", "test"],
        [
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            datamodule.test_dataloader(),
        ],
    ):
        batch = next(iter(loader))
        log.info(f"Batch shapes for {split}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                log.info(f"  {key}: {value.shape}")
            else:
                log.info(f"  {key}: Non-tensor value of type {type(value)}")

    evaluators = {}
    for s in ["train", "val", "test"]:

        inverse_transform_required = cfg.evaluator[s].get(
            "apply_inverse_scaling", False
        )
        if inverse_transform_required:
            inverse_transform_name = cfg.evaluator[s].get(
                "inverse_transform_name", None
            )

            assert inverse_transform_name is not None, (
                f"Evaluator for {s} requires inverse scaling but no "
                "inverse_transform_name is provided. Consider setting the inverse_transform_name in evaluator config group."
            )
        else:
            inverse_transform_name = None

        if inverse_transform_required and inverse_transform_name is None:
            raise ValueError(
                f"Evaluator for {s} requires inverse scaling but no "
                "inverse_transform_name is provided."
            )
        if inverse_transform_name is not None:
            inverse_transform = transforms_manager.get_transform(inverse_transform_name)
            inverse_transform = inverse_transform.transform_instance

            assert isinstance(inverse_transform, InverseTransformMixin), (
                f"The transform {inverse_transform_name} does not implement "
                "InverseTransformMixin but is selected for inverse_transform in evaluator."
            )
        else:
            inverse_transform = None

        remote_logger = None
        for lgr in logger:
            if isinstance(lgr, WandbLogger):
                remote_logger = lgr

        evaluator = hydra.utils.instantiate(
            cfg.evaluator[s],
            inverse_transform=inverse_transform,
            remote_logger=remote_logger,
        )
        log.info(f"Evaluator for {s}: {cfg.evaluator[s]._target_}")
        evaluators[s] = evaluator

    # Instantiate optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    log.info(f"Optimizer: {cfg.optimizer._target_}")

    # Instantiate loss function
    loss = hydra.utils.instantiate(cfg.loss)
    log.info(f"Loss: {cfg.loss._target_}")

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Check if there is ModelCheckpointWithConfig and is so, set the cfg to self.config
    for callback in callbacks:
        if isinstance(callback, ModelCheckpointWithConfig):
            callback.config = cfg

    # TODO: why here? why not at the top of the file?
    # This allows configs to infer data dims from data
    # by using a resolver like ${infer_data_dim:input_key,dim}
    register_data_dim_resolver(data=data_dict["train"])

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
                optimizer=optimizer,
            )
        elif isinstance(backbone_wrapper, AbstractFeedForwardTrainingWrapper):
            model = TrainingLightningModule(
                backbone=backbone_wrapper,
                loss=loss,
                evaluators=evaluators,
                optimizer=optimizer,
            )

        elif isinstance(backbone_wrapper, AbstractFitPredictWrapper):
            model = FitPredictWrapperLightningModule(
                backbone=backbone_wrapper,
                loss=loss,
                evaluators=evaluators,
                optimizer=optimizer,
                debug=cfg.get("debug", False),
            )
        else:
            raise ValueError(
                f"Unsupported backbone wrapper type: {type(backbone_wrapper)}"
            )

    else:
        if cfg.model.metadata.get("lightning_model", False):
            # Model is already a standard LightningModule and does not require a wrapper
            model = hydra.utils.instantiate(cfg.model, evaluators=evaluators)
    model_summary = get_model_summary(model)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
    )

    # This is needed because wandb otherwise only tracks part of the config
    cfg_to_wandb = OmegaConf.to_container(cfg, resolve=True)
    cfg_to_wandb["hydra_cfg"] = OmegaConf.to_container(
        # cannot resolve because not all interpolations are covered by us for now
        HydraConfig.get(),
        resolve=False,
    )

    for lgr in logger:
        if isinstance(lgr, WandbLogger):

            exp_name = get_hydra_override("experiment")
            cfg_to_wandb["exp_name"] = exp_name
            lgr.experiment.config.update(cfg_to_wandb)

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(
            {
                "cfg": cfg,
                "trainer": trainer,
                "model/params/total": model_summary[-1]["total_params"],
                "model/params/trainable": model_summary[-1]["trainable_params"],
            }
        )

    if cfg.task_definition.get("requires_training", False):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

    trainer.test(model=model, datamodule=datamodule)

    return None, None  # metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="run.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    float | None
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
