import importlib
import inspect
import os
from pathlib import Path
import traceback
import torch
import datetime
import hashlib
import torch.nn as nn

import hydra
from hydra import initialize as hydra_initialize, compose
from omegaconf import OmegaConf
from omegaconf import DictConfig
from argparse import ArgumentParser

from lightning.pytorch import  LightningDataModule, LightningModule

from lmetk.utils import set_reproducibility, EarlyStopping
from lmetk.training_strategy.timeseries_transformer import TrainLSTM, TrainSTF
import lmetk.utils.distutil as utils
from lmetk.utils.distutil import auto_load_model
from lmetk.training_strategy.timeseries_transformer_validation import extended_validation

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

def create_parser():
    parser = ArgumentParser()
    
    # pipelne specific options, not related to the task
    parser.add_argument("--test_only", action="store_true")
    
    parser.add_argument("--working_dir", type=str, default=None, help="working directory")
    
    parser.add_argument("--exp_name", default=None, type=str, help="experiment name")
    parser.add_argument("--resume", default=None, type=str, help="reload run from directory")
    
    return parser


def hydra_load(overrides):
    with hydra_initialize(version_base="1.2", config_path="configs"):
        cfg = compose(config_name="main.yaml", overrides=overrides)
        return cfg
    
    
def flatten_keys(config, parent_key='', separator='.'):
    keys = []
    for k, v in config.items():
        full_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            keys.extend(flatten_keys(v, full_key, separator))
        else:
            keys.append(full_key)
    return keys


def get_class_defaults(class_str: str):
    
    splits = class_str.split('.')
    module_str = ".".join(splits[:-1])
    module = importlib.import_module(module_str)
    clazz = getattr(module, splits[-1])
    h = clazz.__init__
    
    spec = inspect.getfullargspec(h)
    keys = reversed(spec.args)
    values = reversed(spec.defaults)
    return OmegaConf.create({k: v for k, v in zip(keys, values)})


def validate_and_merge_configs(base_config, update_config):
    return OmegaConf.merge(base_config, update_config)


def configure(exp_name, working_dir=None) -> DictConfig:
    
    working_dir = str(Path(working_dir).expanduser()) if working_dir is not None else None
    
    cfg: DictConfig = hydra_load([f"experiment={exp_name}"])
    OmegaConf.set_struct(cfg, False)
    flags = OmegaConf.load("flags.yaml")
    
    if flags.trainer.epochs is None:
        flags.trainer.epochs = cfg.trainer.epochs
        
    if flags.trainer.batch_size is None:
        flags.trainer.batch_size = cfg.trainer.batch_size
    
    cfg = OmegaConf.merge(cfg, flags)
    
    if "_target_" in cfg.model:
        model_defaults = get_class_defaults(cfg.model._target_)
        cfg.model = OmegaConf.merge(model_defaults, cfg.model)
        
    if "_target_" in cfg.datamodule:
        dm_defaults = get_class_defaults(cfg.datamodule._target_)
        cfg.datamodule = OmegaConf.merge(dm_defaults, cfg.datamodule)
    
    working_dir = "working_dir" if working_dir is None else working_dir
    
    now = datetime.datetime.now()
    rcode = f"{now.year}_{now.month}_{now.day}_{now.hour}-{now.minute}-{now.second}"
    prm_hash = hashlib.md5(str(cfg).encode('utf-8')).hexdigest()
    rnd_name = f"R_{rcode}_{exp_name}_seed{cfg.glob.seed}_{prm_hash}"
    
    if cfg.info.exp_name is None:
        cfg.info.exp_name = f"{exp_name}_{rcode}_seed{cfg.glob.seed}_{prm_hash}"
    
    rnd_dir = f"{working_dir}/runs/{rnd_name}/"
    out_dir = rnd_dir + "outputs/"
    ckp_dir = rnd_dir + "checkpoints/"
    log_dir = rnd_dir + "logs/"
    best_ckp_dir = rnd_dir + "best_checkpoints"
    
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(out_dir)
    os.makedirs(ckp_dir)
    os.makedirs(log_dir)
    os.makedirs(best_ckp_dir)
    
    dcfg = OmegaConf.create({
        "dirs": {
            "rnd_dir": rnd_dir,
            "out_dir": out_dir,
            "ckp_dir": ckp_dir,
            "log_dir": log_dir,
            "best_ckp_dir": best_ckp_dir,
            'runtime_dir': os.getcwd(),
            "working_dir": working_dir,
        }
    })
    
    cfg = OmegaConf.merge(cfg, dcfg)
    OmegaConf.set_struct(cfg, True)
    return cfg
    
    
    
def initialize(cfg):
    
    device = torch.device(cfg.glob.device)
    set_reproducibility(cfg.glob.seed)
    
    if "num_threads" in cfg.trainer and cfg.trainer.num_threads is not None:
        torch.set_num_threads(cfg.trainer.num_threads)
    
    dm: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup()
    
    if cfg.glob.distributed:
        raise Exception("Distributed training is not supported yet!")
    else:
        global_rank = 0
        if cfg.lightning.use_lightning_dataloader:
            data_loader_train = dm.train_dataloader()
            data_loader_val = dm.val_dataloader()
            data_loader_test = dm.test_dataloader()
        else:
            raise Exception("Not implemented yet!")

    if global_rank == 0 and cfg.dirs.log_dir is not None:
        os.makedirs(cfg.dirs.log_dir, exist_ok=True)
        

    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if cfg.glob.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[cfg.distributed_args.gpu], 
            find_unused_parameters=True)
        model_without_ddp = model.module
    elif cfg.glob.multi_gpu:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)
        
    
    if cfg.lightning.use_lightning_optimizer:
        
        ocfg = model.configure_optimizers()
        
        if isinstance(ocfg, tuple):
            optimizer, scheduler = ocfg
            optimizer = optimizer[0]
            scheduler = scheduler[0]
        elif isinstance(ocfg, dict):
            optimizer = ocfg['optimizer']
            scheduler = ocfg['lr_scheduler']['scheduler']
    
        lr = optimizer.param_groups[0]["lr"]
        loss_scaler = None
        lr_schedule_values = None
        wd_schedule_values = None
            
    else:
        raise Exception("Not implemented yet!")

    model.to(device)
    model_without_ddp = model
    
    loading_success = auto_load_model(
        cfg=cfg, 
        auto_resume=cfg.reload.auto_resume,
        model=model, 
        model_without_ddp=model_without_ddp, 
        optimizer=optimizer, 
        scheduler=scheduler,
        loss_scaler=loss_scaler,
        model_ema=None,
        device=device,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg.info.n_parameters = n_parameters
    
    early_stopping = EarlyStopping(patience=cfg.trainer.early_stopping.patience, verbose=True)
    
    if loading_success:
        best_metrics_dict = OmegaConf.load(cfg.dirs.best_ckp_dir + "/best_metrics.yaml")
    else:
        best_metrics_dict = {
            "val_loss":[10.0e12, -1, -1],
        }

    objects = dict(
        train_data_loader=data_loader_train,
        val_data_loader=data_loader_val,
        test_data_loader=data_loader_test,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        scheduler=scheduler,
        best_metrics_dict = best_metrics_dict,
        lr_schedule_values=lr_schedule_values,
        wd_schedule_values=wd_schedule_values,
        device = device,
        datamodule=dm,
        early_stopping=early_stopping
    )
    
    if cfg.trainer._target_ == "TrainSTF":
        trainer = TrainSTF(cfg, objects)
    elif cfg.trainer._target_ == "TrainLSTM":
        trainer = TrainLSTM(cfg, objects)
    
    OmegaConf.save(cfg, cfg.dirs.rnd_dir +'/config.yaml')
    return trainer
    
    
if __name__ == "__main__":
    
    try:
        parser = create_parser()
        args = parser.parse_args()
        if args.resume is not None:
            cfg = OmegaConf.load(args.resume + '/config.yaml')
            cfg.reload.auto_resume = True
        else:
            cfg = configure(args.exp_name, working_dir=args.working_dir)
            
        trainer = initialize(cfg)
        run_evaluation = True
        if not args.test_only:  
            trainer.train()
            
    except KeyboardInterrupt:
        print("Manually interrupting epoch loop (KeyboardInterrupt) ...\n")

    except:
        print(traceback.format_exc())
        
    finally:
        dm_class = cfg['datamodule']['_target_'].split(".")[-1]        
        if run_evaluation:
            best_metrics_dict = trainer.objects['best_metrics_dict']
            trainer.test(best_metrics_dict)
            if (dm_class == "TVTContextDataModule" or dm_class == "SpatialDataModule"):
                extended_validation(Path(cfg.dirs.rnd_dir))