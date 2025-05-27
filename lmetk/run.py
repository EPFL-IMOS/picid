"""Main entry point for training and testing models."""

import os
import random

# shutil
from typing import Any

import hydra
import lightning as L
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from lmetk.utils_refactor import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)



def initialize_hydra() -> DictConfig:
    """Initialize Hydra when main is not an option (e.g. tests).

    Returns
    -------
    DictConfig
        A DictConfig object containing the config tree.
    """
    hydra.initialize(
        version_base="1.3", config_path="../configs", job_name="run"
    )
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

    # Instantiate and load dataset
    log.info(f"Instantiating loader <{cfg.datasource.loader._target_}>")
    dataset_source = hydra.utils.instantiate(cfg.datasource.loader)
    dataset_source._pipeline()
    

    # Get data
    features, time_features = dataset_source.get_data()

    # Prepare datasets
    dataset_train = hydra.utils.instantiate(
        cfg.dataset,
        features=features["train"], 
        time_features=time_features["train"]
        )

    dataset_val = hydra.utils.instantiate(
        cfg.dataset,
        features=features["val"], 
        time_features=time_features["val"]
        )
    
    dataset_test = hydra.utils.instantiate(
            cfg.dataset,
            features=features["test"], 
            time_features=time_features["test"]
            )
    print(f"Train dataset size: {dataset_train.__len__()}")
    print(f"Validation dataset size: {dataset_val.__len__()}")
    print(f"Test dataset size: {dataset_test.__len__()}")


    evaluator = hydra.utils.instantiate(cfg.evaluator)
    print(f"Evaluator: {cfg.evaluator._target_}")

    # Instantiate optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    print(f"Optimizer: {cfg.optimizer._target_}")
    
    # Instantiate loss function
    loss = hydra.utils.instantiate(cfg.loss)
    print(f"Loss: {cfg.loss._target_}")

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))


    # model: LightningModule = hydra.utils.instantiate(
    #     cfg.model,
    #     evaluator=cfg.evaluator,
    #     optimizer=cfg.optimizer,
    #     loss=cfg.loss,
    # )
    

    # # Preprocess dataset and load the splits
    # log.info("Instantiating preprocessor...")
    # transform_config = cfg.get("transforms", None)
    # preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    # dataset_train, dataset_val, dataset_test = (
    #     preprocessor.load_dataset_splits(cfg.dataset.split_params)
    # )
    # # Prepare datamodule
    # log.info("Instantiating datamodule...")
    # if cfg.dataset.parameters.task_level in ["node", "graph"]:
    #     datamodule = TBDataloader(
    #         dataset_train=dataset_train,
    #         dataset_val=dataset_val,
    #         dataset_test=dataset_test,
    #         **cfg.dataset.get("dataloader_params", {}),
    #     )
    # else:
    #     raise ValueError("Invalid task_level")

    # # Model for us is Network + logic: inputs backbone, readout, losses
    # log.info(f"Instantiating model <{cfg.model._target_}>")
    # model: LightningModule = hydra.utils.instantiate(
    #     cfg.model,
    #     evaluator=cfg.evaluator,
    #     optimizer=cfg.optimizer,
    #     loss=cfg.loss,
    # )

    # log.info("Instantiating callbacks...")
    # callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # log.info("Instantiating loggers...")
    # logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # # Log to wandb preprocessor time
    # if logger:
    #     for log_temp in logger:
    #         if isinstance(log_temp, L.pytorch.loggers.wandb.WandbLogger):
    #             log_temp.log_metrics(
    #                 {
    #                     "preprocessor_time": preprocessor.preprocessing_time,
    #                 }
    #             )

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(
    #     cfg.trainer,
    #     callbacks=callbacks,
    #     logger=logger,
    #     num_sanity_val_steps=0,
    #     log_every_n_steps=0,  # Avoid console logging
    # )

    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }

    # if logger:
    #     log.info("Logging hyperparameters!")
    #     log_hyperparameters(object_dict)

    # if cfg.get("train"):
    #     log.info("Starting training!")
    #     trainer.fit(
    #         model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    #     )

    # train_metrics = trainer.callback_metrics
    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     test_best_model_path = True
    #     if cfg.get("ckpt_path"):
    #         ckpt_path = cfg.ckpt_path
    #         log.info(
    #             f"Attempting to load weights from the provided ckpt_path: {ckpt_path}"
    #         )
    #         try:
    #             trainer.test(
    #                 model=model, datamodule=datamodule, ckpt_path=ckpt_path
    #             )
    #             test_best_model_path = False  # do not test "best model" if a valid ckpt_path is provided
    #         except FileNotFoundError:
    #             log.warning(
    #                 f"No checkpoint file found at the provided ckpt_path: {ckpt_path}."
    #             )
    #             log.info("Trying with best model instead...")
    #     if test_best_model_path:
    #         ckpt_path = trainer.checkpoint_callback.best_model_path
    #         if ckpt_path == "":
    #             log.warning(
    #                 "Best ckpt not found! Using current weights for testing..."
    #             )
    #             ckpt_path = None
    #         trainer.test(
    #             model=model, datamodule=datamodule, ckpt_path=ckpt_path
    #         )

    #     # Remove saved model (useful with sweeps)
    #     # Check if the file exists and delete it
    #     if os.path.exists(ckpt_path):
    #         os.remove(ckpt_path)

    # test_metrics = trainer.callback_metrics

    # # Merge train and test metrics
    # metric_dict = {**train_metrics, **test_metrics}

    return None, None # metric_dict, object_dict


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
        num_params: int = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    else:  # counts trainable and none-traibale
        num_params: int = sum(p.numel() for p in model.parameters() if p)
    assert num_params > 0, f"Err: {num_params=}"
    return int(num_params)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="run.yaml"
)
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
