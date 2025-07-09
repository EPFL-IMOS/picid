"""Main entry point for training and testing models."""

import random
import torch
# shutil
from typing import Any
from pathlib import Path
import hydra
import lightning as L
import numpy as np
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from lmetk.data.preprocessing.single_source_preprocessor import PreProcessor
from lmetk.data.preprocessing.multi_source_preprocessor import MultiSourcePreProcessor
from lmetk.data.preprocessing.per_source_multi_source_preprocessor import PerSourceMultiSourcePreProcessor
from box import Box
from lightning.pytorch.loggers.wandb import WandbLogger


from lmetk.utils_refactor import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)

from rich.table import Table
from rich.console import Console


from lmetk.pipeline.base import ConstantLossLightningModule
from lmetk.transforms.transform_manager import ConfigTransformManager


from lmetk.data.datasources import DataSourceLoader
from lmetk.data.datasources.multi_source_loader import MultiSourceLoader    
from lmetk.data.datasources.phmd_battery import PHMDMultiSourceLoader

def get_hydra_override(key: str):
    if not HydraConfig.initialized():
        return None
    for override in HydraConfig.get().overrides.task:
        if override.startswith(f"{key}="):
            return override.split("=", 1)[1]
    return None


def extract_targets(cfg, prefix=""):
    targets = []
    if not OmegaConf.is_dict(cfg):
        return targets

    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if OmegaConf.is_dict(v):
            if "_target_" in v:
                target_str = v["_target_"]
                if "." in target_str and not target_str.endswith(".py"):
                    # Try to resolve to a Python file (best effort)
                    module_path = Path(target_str.replace(".", "/") + ".py").resolve().as_uri()
                else:
                    module_path = ""
                targets.append((full_key, target_str, module_path))
            # recurse into nested dict
            targets.extend(extract_targets(v, prefix=full_key))
    return targets

def display_targets(cfg):
    table = Table(title="Hydra _target_ Classes")
    table.add_column("Config Path", style="cyan", no_wrap=True)
    table.add_column("_target_", style="magenta")
    table.add_column("Resolved Module Path", style="green", overflow="fold", no_wrap=False)


    for path, target, resolved in extract_targets(cfg):
        table.add_row(path, target, resolved)

    Console().print(table)


def get_config_info(cfg):
    overrides = HydraConfig.get().overrides.task
    defaults = cfg.defaults if 'defaults' in cfg else []

    # 1. Build a mapping of override keys (from CLI or config)
    override_map = {}
    for item in overrides:
        if '=' in item:
            k, v = item.split('=', 1)
            override_map[k] = f"CLI ({item})"

    # 2. Traverse defaults chain to see where each group came from
    default_sources = {}
    for d in defaults:
        if isinstance(d, dict):
            for k, v in d.items():
                if v is not None:
                    default_sources[k] = f"{k}/{v}.yaml"
        elif isinstance(d, str):
            name = d.split("/")[0]
            default_sources[name] = f"{d}.yaml"

    # 3. Traverse config tree
    table_data = []
    def recurse(node, prefix=""):
        if not OmegaConf.is_dict(node):
            return
        for k, v in node.items():
            key_path = f"{prefix}.{k}" if prefix else k
            if OmegaConf.is_dict(v):
                recurse(v, prefix=key_path)
            else:
                # Who set it?
                override = override_map.get(key_path, default_sources.get(k, ""))
                source = getattr(cfg._metadata.config_sources.get(k, {}), "path", "")
                table_data.append((key_path, str(v), override or "default", source or ""))
    recurse(cfg)
    return table_data


def display_config_sources(cfg):
    data = get_config_info(cfg)
    table = Table(title="Hydra Config Resolution")
    table.add_column("Key", style="cyan")
    table.add_column("Final Value", style="magenta")
    table.add_column("Overridden By", style="green")
    table.add_column("Source Config", style="yellow")

    for row in data:
        table.add_row(*row)

    Console(width=180).print(table)

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

    for k, d in cfg.paths.items():
        log.info(f"Directory for {k} is: {Path(d).expanduser().resolve().as_uri()}")

    # HydraConfig.set_config(cfg)
    # display_config_sources(cfg)
    display_targets(cfg)

    # Instantiate and load dataset
    log.info(f"Instantiating loader <{cfg.datasource._target_}>")
    dataset_source = hydra.utils.instantiate(cfg.datasource)
    # Create manager
    
    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    # Get transform names
    log.info("Transform names:", transforms_manager.get_transform_names())
    
    # Get transforms that apply to features
    feature_transforms = transforms_manager.get_transforms_by_apply_to("features")
    log.info("Feature transforms:", list(feature_transforms.keys()))
    
    # Get transforms with auxiliary keys
    aux_transforms = transforms_manager.get_transforms_with_auxiliary_keys()
    log.info("Transforms with auxiliary keys:", list(aux_transforms.keys()))
    
    if isinstance(dataset_source, DataSourceLoader):
        preprocessor = PreProcessor(dataset_source, transforms_manager, logger=log)
    
    elif isinstance(dataset_source, MultiSourceLoader):
        preprocessor = MultiSourcePreProcessor(dataset_source, transforms_manager, logger=log)
   
    elif isinstance(dataset_source, PHMDMultiSourceLoader):
        preprocessor = PerSourceMultiSourcePreProcessor(dataset_source, transforms_manager, logger=log)
    
    else:
        raise ValueError("Invalid dataset source type")
    
    preprocessor.pipeline()

    data_dict = preprocessor.get_processed_data()
    
    if dataset_source.get_data_name() == 'railway':
        scaler = transforms_manager.get_data_transform('scaler_target').transform_instance.scaler
    else: 
        scaler = lambda x: x

    
    # Extract additional inputs to dataset if needed.
    add_args = {}
    if "N-CMAPSS" in cfg.datasource.data_name:
        dataset_train = hydra.utils.instantiate(cfg.dataset, data_dict=data_dict, split='train')
        
        #TODO: For now we are not working with the validation, only test
        dataset_val = hydra.utils.instantiate(cfg.dataset, data_dict=data_dict, split='test')
        
        dataset_test = hydra.utils.instantiate(cfg.dataset, data_dict=data_dict, split='test')

    elif cfg.datasource.data_name == 'railway' and cfg.datasource.mode_mode == 'context':
            # Prepare datasets
            dataset_train = hydra.utils.instantiate(
                cfg.dataset,
                context=data_dict['features']["train"],
                target= data_dict['target']["train"],
                time_features=data_dict['timestamps']["train"],
                
                )

            dataset_val = hydra.utils.instantiate(
                cfg.dataset,
                context=data_dict['features']["val"], 
                target= data_dict['target']["val"],
                time_features=data_dict['timestamps']["val"],
                
                )
            
            dataset_test = hydra.utils.instantiate(
                    cfg.dataset,
                    context=data_dict['features']["test"], 
                    target= data_dict['target']["test"],
                    time_features=data_dict['timestamps']["test"],
                    
                    )
    elif cfg.datasource.data_name in ['NB1', 'NB14','UNIBO21']:
        dataset_train = hydra.utils.instantiate(cfg.dataset, data_dict=data_dict, split='train')
        dataset_val = hydra.utils.instantiate(cfg.dataset, data_dict=data_dict, split='val')
        dataset_test = hydra.utils.instantiate(cfg.dataset, data_dict=data_dict, split='test')
    else:
        assert cfg.datasource.mode_mode == 'other', " Ban naming of the variable (datasource.mode_mode). This one basically help to identify which are the inputs of the dataset object. context means features and targets, while others means features (where one of the features is actually a target) and timestamps"
        # Prepare datasets
        dataset_train = hydra.utils.instantiate(
            cfg.dataset,
            features=data_dict['features']["train"], 
            time_features=data_dict['timestamps']["train"],
            **add_args
            )

        dataset_val = hydra.utils.instantiate(
            cfg.dataset,
            features=data_dict['features']["val"], 
            time_features=data_dict['timestamps']["val"],
            **add_args
            )
        
        dataset_test = hydra.utils.instantiate(
                cfg.dataset,
                features=data_dict['features']["test"], 
                time_features=data_dict['timestamps']["test"],
                **add_args
                )
    log.info(f"Train dataset size: {dataset_train.__len__()}")
    log.info(f"Validation dataset size: {dataset_val.__len__()}")
    log.info(f"Test dataset size: {dataset_test.__len__()}")

    # Instantiate dataloader
    datamodule = hydra.utils.instantiate(cfg.datamodule, 
        dataset_train = dataset_train,
        dataset_val = dataset_val,
        dataset_test = dataset_test,
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

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    if not cfg.model.get('metadata', False):
        add_backbone_args = {}
        if cfg.datasource.data_name == 'railway':
            add_backbone_args['train_dataloader'] = datamodule.train_dataloader()
            add_backbone_args['val_dataloader'] = datamodule.val_dataloader()

        backbone_wrapper = hydra.utils.instantiate(cfg.model, **add_backbone_args) # TimePFNWrapper(backbone, **cfg.model)
        
        if cfg.model.get('load_path', False):
            backbone_wrapper.backbone.load_state_dict(torch.load(cfg.model.load_path))
        
        model = ConstantLossLightningModule(
            backbone = backbone_wrapper,
            loss=loss,
            evaluators=evaluators,
            optimizer=optimizer,
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

    # has to happen after trainer is instantiated
    for lgr in logger:
        if isinstance(lgr, WandbLogger):
            cfg_to_wandb = OmegaConf.to_container(cfg, resolve=True)
            cfg_to_wandb["hydra_cfg"] = OmegaConf.to_container(
                # cannot resolve because not all interpolations are covered by us for now
                HydraConfig.get(), resolve=False
            )
            exp_name = get_hydra_override("experiment")
            cfg_to_wandb["exp_name"] = exp_name
            lgr.experiment.config.update(cfg_to_wandb)


    if cfg.task_definition.get("requires_training", False):
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule,
        )
    
    trainer.test(model=model, datamodule=datamodule)

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
