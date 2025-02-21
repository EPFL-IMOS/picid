import datetime
import hashlib
import os
import hydra
from hydra import initialize as hydra_initialize, compose
from omegaconf import DictConfig, OmegaConf

from lmetk.exp.railway_traction import RailwayTractionExperiment


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
    
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(out_dir)
    os.makedirs(plt_dir)
    os.makedirs(mod_dir)
    os.makedirs(ckp_dir)
    os.makedirs(log_dir)
    os.makedirs(best_ckp_dir)
    
    return cfg
    
    
def initialize(cfg):
    
    dm = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    
    objs = {
        "model": model,
        "dm": dm
    }
    
    exp = RailwayTractionExperiment(cfg, objs)
    
    return exp