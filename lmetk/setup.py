import datetime
import hashlib
import os
import hydra
from hydra import initialize as hydra_initialize, compose
from omegaconf import DictConfig, OmegaConf

from lmetk.utils.logging import MultiLogger


def hydra_load(overrides):
    with hydra_initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(config_name="main.yaml", overrides=overrides)
        return cfg


def configure(exp_name, data_dir=None, working_dir=None) -> DictConfig:
    cfg: DictConfig = hydra_load([f"experiment={exp_name}"])
    working_dir = "testwdir" if working_dir is None else working_dir
    
    now = datetime.datetime.now()
    rcode = f"{now.year}_{now.month}_{now.day}_{now.hour}-{now.minute}-{now.second}"
    # rnd_dir = f"{args.working_dir}/runs/seed{reproducible}_as{params['steps']}_R{rcode}/"
    prm_hash = hashlib.md5(str(cfg).encode('utf-8')).hexdigest()
    rnd_name = f"R_{rcode}_{exp_name}_seed{cfg.glob.seed}_{prm_hash}"
    
    if cfg.info.exp_name is None:
        cfg.info.exp_name = f"{exp_name}_{rcode}_seed{cfg.glob.seed}_{prm_hash}"
    
    rnd_dir = f"{working_dir}/runs/{rnd_name}/"
    out_dir = rnd_dir + "outputs/"
    plt_dir = rnd_dir + "plots/"
    mod_dir = rnd_dir + "models/"
    ckp_dir = rnd_dir + "checkpoints/"
    log_dir = rnd_dir + "logs/"
    best_ckp_dir = rnd_dir + "best_checkpoints"
    
    optuna_out_dir = f"{working_dir}/optuna"
    wandb_out_dir = f"{working_dir}/wandb"
    tensorboard_out_dir = f"{working_dir}/tensorboard/{rnd_name}"
    
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(optuna_out_dir, exist_ok=True)
    os.makedirs(wandb_out_dir, exist_ok=True)
    os.makedirs(tensorboard_out_dir, exist_ok=True)
    os.makedirs(out_dir)
    os.makedirs(plt_dir)
    os.makedirs(mod_dir)
    os.makedirs(ckp_dir)
    os.makedirs(log_dir)
    os.makedirs(best_ckp_dir)
    
    dcfg = OmegaConf.create({
        "dirs": {
            "rnd_dir": rnd_dir,
            "plt_dir": plt_dir,
            "mod_dir": mod_dir,
            "out_dir": out_dir,
            "ckp_dir": ckp_dir,
            "log_dir": log_dir,
            "best_ckp_dir": best_ckp_dir,
            "optuna_out_dir": optuna_out_dir,
            "wandb_out_dir": wandb_out_dir,
            "tensorboard_out_dir": tensorboard_out_dir,
            'runtime_dir': os.getcwd(),
            "working_dir": working_dir,
            "dataset_dir": data_dir
        }
    })
    
    OmegaConf.set_struct(cfg, False) # enable attribute access
    cfg = OmegaConf.merge(cfg, dcfg)
    OmegaConf.set_struct(cfg, True) # enable attribute access
    print(cfg)
    
    print(f"RUN: {rnd_dir}\n")   
    return cfg
    
    
    
def initialize(cfg):
    
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup()
    
    model = hydra.utils.instantiate(cfg.model)

    if cfg.logging.wandb_enabled and cfg.reload.wandb_id is not None:
        wandb_id = cfg.reload.wandb_id
    else:
        wandb_id = None

    log_writer = MultiLogger(
        exp_name=cfg.info.exp_name,
        project_name=cfg.info.project_name,
        entity_name=cfg.info.entity_name,
        config=OmegaConf.to_container(cfg, resolve=True), 
        logdir_wandb=cfg.dirs.wandb_out_dir,
        logdir_tb=cfg.dirs.tensorboard_out_dir,
        wandb_enabled=cfg.logging.wandb_enabled,
        wandb_resume=cfg.reload.wandb_id is not None,
        wandb_id=wandb_id)
    
    objs = {
        "model": model,
        "dm": dm,
        "log_writer": log_writer
    }
    
    exp_factory = hydra.utils.instantiate(cfg.training_strategy, _partial_=True)
    exp = exp_factory(cfg=cfg, objs=objs)
    
    return exp