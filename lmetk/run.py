"""Main entry point for training and testing models."""

import random
from pathlib import Path

# shutil
from typing import Any

import hydra
import lightning as L
import numpy as np
import torch

from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from lmetk.data.datasources import SingleSourceLoader
from lmetk.data.datasources.base.multi_source_loader import MultiSourceLoader
from lmetk.data.datasources.base.phmd_loader import PHMDMultiSourceLoader
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
from lmetk.transforms.base.transform_manager import ConfigTransformManager

# from lmetk.utils.rich_output import display_targets, print_data_dict_structure
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
    # Set seed for random number generators in pytorch, numpy and python.random
    L.seed_everything(cfg.seed, workers=True)
    # Seed for torch
    torch.manual_seed(cfg.seed)
    # Seed for numpy
    np.random.seed(cfg.seed)
    # Seed for python random
    random.seed(cfg.seed)

    for k, d in cfg.paths.items():
        log.info(f"Directory for {k} is: {Path(d).expanduser().resolve().as_uri()}")

    # HydraConfig.set_config(cfg)
    # display_config_sources(cfg)
    display_targets(cfg)

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
    feature_transforms = transforms_manager.get_transforms_by_apply_to("features")
    log.info("Feature transforms:" + str(list(feature_transforms.keys())))

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
    preprocessor.pipeline()

    data_dict = preprocessor.get_processed_data_dict(return_splits_on_first_level=True)
    meta_data_dict = preprocessor.get_meta_data_dict()

    assert set(data_dict.keys()) == {"train", "val", "test"}

    features_set = set(data_dict["train"].keys())
    for split in ["val", "test"]:
        assert (
            set(data_dict[split].keys()) == features_set
        ), f"Mismatch in features for split {split}. Expected: {features_set}, Found: {set(data_dict[split].keys())}"

    if cfg.datasource.data_name in ["NB1", "NB14", "UNIBO21", "railway"]:
        scaler = transforms_manager.get_data_transform(
            "scaler_target"
        ).transform_instance.scaler

    else:
        scaler = lambda x: x

    # Extract additional inputs to dataset if needed.
    dataset_train = hydra.utils.instantiate(
        cfg.dataset, data_dict=data_dict["train"], meta_data_dict=meta_data_dict
    )
    dataset_val = hydra.utils.instantiate(
        cfg.dataset, data_dict=data_dict["test"], meta_data_dict=meta_data_dict
    )
    dataset_test = hydra.utils.instantiate(
        cfg.dataset, data_dict=data_dict["test"], meta_data_dict=meta_data_dict
    )

    log.info(f"Train dataset size: {dataset_train.__len__()}")
    log.info(f"Validation dataset size: {dataset_val.__len__()}")
    log.info(f"Test dataset size: {dataset_test.__len__()}")

    # Instantiate dataloader
    datamodule = hydra.utils.instantiate(
        cfg.datamodule,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
    )

    evaluators = {}
    for s in ["train", "val", "test"]:
        evaluator = hydra.utils.instantiate(cfg.evaluator[s], scaler=scaler)
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

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    if not cfg.model.get("metadata", False):
        add_backbone_args = {}
        if cfg.datasource.data_name == "railway":
            add_backbone_args["train_dataloader"] = datamodule.train_dataloader()
            add_backbone_args["val_dataloader"] = datamodule.val_dataloader()

        backbone_wrapper = hydra.utils.instantiate(
            cfg.model, **add_backbone_args
        )  # TimePFNWrapper(backbone, **cfg.model)

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
            )
        else:
            raise ValueError(
                f"Unsupported backbone wrapper type: {type(backbone_wrapper)}"
            )

    else:
        if cfg.model.metadata.get("lightning_model", False):
            # Model is already a standard LightningModule and does not require a wrapper
            model = hydra.utils.instantiate(cfg.model, evaluators=evaluators)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
    )

    # TODO: Do we need this? log_hyperparameters already should be logging cfg to wandb
    # has to happen after trainer is instantiated
    # for lgr in logger:
    #     if isinstance(lgr, WandbLogger):
    #         cfg_to_wandb = OmegaConf.to_container(cfg, resolve=True)
    #         cfg_to_wandb["hydra_cfg"] = OmegaConf.to_container(
    #             # cannot resolve because not all interpolations are covered by us for now
    #             HydraConfig.get(),
    #             resolve=False,
    #         )
    #         exp_name = get_hydra_override("experiment")
    #         cfg_to_wandb["exp_name"] = exp_name
    #         lgr.experiment.config.update(cfg_to_wandb)

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters({"cfg": cfg, "trainer": trainer})

    if cfg.task_definition.get("requires_training", False):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

    trainer.test(model=model, datamodule=datamodule)

    return None, None  # metric_dict, object_dict


def count_number_of_parameters(
    model: torch.nn.Module, only_trainable: bool = True
) -> int:
    """Count the number of trainable params.

    If all params, specify only_trainable = False.

    Ref:
        - https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?u=brando_miranda
        - https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model/62764464#62764464

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    only_trainable : bool, optional
        If True, only count trainable parameters (default: True).

    Returns
    -------
    int
        The number of parameters.
    """
    if only_trainable:
        num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:  # counts trainable and none-traibale
        num_params: int = sum(p.numel() for p in model.parameters() if p)
    assert num_params > 0, f"Err: {num_params=}"
    return int(num_params)


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
