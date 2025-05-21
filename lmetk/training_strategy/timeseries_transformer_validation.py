import torch
import pandas as pd
import numpy as np
import hydra

from einops import rearrange
from omegaconf import OmegaConf

import utils.eval_stats as stats



class ModelEvaluationContainerNoLightning:
    
    def __init__(self, cfg, checkpoint_path: str, DEVICE:str) -> None:
        self.cfg = cfg
        
        model = hydra.utils.instantiate(cfg.model)
        
        print(f'Loading model {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)    
        
        model.eval()
        self.__model = model.to(DEVICE)
        self.__cfg = cfg
        
    @property
    def model(self):
        return self.__model   



def eval_mec(mec, N_samples_per_day=24, dataset="test", ts_in=24):    
    gt = []
    fc = []
    
    dm = hydra.utils.instantiate(mec.cfg.datamodule)
    dm.prepare_data()
    dm.setup()
    
    dataset_name = dataset if dataset != "train" else "train_test"
    test_dl = dm._create_dataloader(dm.datasets[dataset_name], batch_size=64, shuffle=False)
    
    if dataset == "test":
        rr = dm.r_test
        dd = dm.test
    elif dataset == "val":
        rr = dm.r_val
        dd = dm.val
    elif dataset == "train":
        rr = dm.r_train
        dd = dm.train
    elif dataset == "test_small":
        rr = dm.r_test_small
        dd = dm.test_small
    else:
        raise ValueError("Dataset should be one of test, val, train")

    eval_kwargs = mec.model.eval_step_forward_kwargs
    for batch in test_dl:
        batch = [b.to('cuda:0') for b in batch]
        _,_,_, F_t = batch
        
        if "target_mask" in eval_kwargs:
            F_t = F_t[:,:,eval_kwargs["target_mask"]]
        
        gt.append(F_t.detach().cpu().numpy()[:, -N_samples_per_day:,:])
        forecast_output, *_ = mec.model.forward_model_pass(*batch, **eval_kwargs)
        fc.append(forecast_output.detach().cpu().numpy())
    
    gt_np = rearrange(np.concatenate(gt), 'D T F -> (D T F)')
    fc_np = rearrange(np.concatenate(fc), 'D T F -> (D T F)')
    
    
    gt_np_rescaled = dm.inverse_scale_bahnlast(gt_np.reshape(-1, 1))
    fc_np_rescaled = dm.inverse_scale_bahnlast(fc_np.reshape(-1, 1))
    
    if mec.cfg.datamodule.trend_decomp and mec.cfg.datamodule.trend_decomp == "mean":
        fc_np_rescaled = fc_np_rescaled[:, 0] + dm.prev_mean_shifted[rr][ts_in:]
        gt_np_rescaled = gt_np_rescaled[:, 0] + dm.prev_mean_shifted[rr][ts_in:]

        gt_np_rescaled = gt_np_rescaled.values
        fc_np_rescaled = fc_np_rescaled.values
    else:
        gt_np_rescaled = gt_np_rescaled[:, 0]
        fc_np_rescaled = fc_np_rescaled[:, 0]
    
    df = pd.DataFrame(dict(
        gt=gt_np,
        gt_true=dm.data['gt'].iloc[rr].values[ts_in:],
        fc=fc_np,
        gt_rescaled=gt_np_rescaled,
        fc_rescaled=fc_np_rescaled), 
        index=dd.index[ts_in:])
    return df


def extended_validation(path, N_samples_per_day = 24, ts_in=None, cfg=None):

    best_metrics_dict = OmegaConf.load(path / "best_checkpoints" / "best_metrics.yaml")
    ckpt_best = path / "best_checkpoints" / f"best_val_{best_metrics_dict['val_loss'][1]}.pth"
    
    if cfg is None:
        cfg = OmegaConf.load(path / "config.yaml")
    
    if ts_in is None:
        ts_in = cfg.datamodule.ts_in
    
    out_dir = path / "outputs"

    mec = ModelEvaluationContainerNoLightning(cfg, ckpt_best, cfg.glob.device)
    dm_class = cfg['datamodule']['_target_'].split(".")[-1]      
    if dm_class == "SpatialDataModule":
        datasets = ['train', 'val', 'test', "test_small"]
    
    elif dm_class ==  "TVTContextDataModule":
        datasets = ["train", "val", "test"]
    else:
        raise(ValueError("Unknown datamodule"))
    
    stats_dfs = []
    for ds in datasets:
        do = out_dir / ds
        do.mkdir(parents=False, exist_ok=True)
        df = eval_mec(mec, N_samples_per_day=N_samples_per_day, dataset=ds, ts_in=ts_in)
        
        scaled_true = df['gt_true'].to_numpy()
        scaled_pred =df['fc_rescaled'].to_numpy()
        norm_true = df['gt'].to_numpy()
        norm_pred = df['fc'].to_numpy()
        
        df.to_csv(do/'full_evaluation.csv', sep=";")
        
        stats_df = {
            "mape": stats.mape(scaled_true, scaled_pred, agg="disabled"),
            "mae": stats.mae(scaled_true, scaled_pred, agg="disabled"),
            "mse": stats.mse(scaled_true, scaled_pred, agg="disabled"),
            "smape": stats.smape(scaled_true, scaled_pred, agg="disabled"),

            "normalized_mape": stats.mape(norm_true, norm_pred, agg="disabled"),
            "normalized_mae": stats.mae(norm_true, norm_pred, agg="disabled"),
            "normalized_mse": stats.mse(norm_true, norm_pred, agg="disabled"),
            "normalized_smape": stats.smape(norm_true, norm_pred, agg="disabled"),
        }
        
        stats_df = pd.DataFrame(stats_df, index=df.index)
        stats_df.to_csv(do/'full_stats.csv', sep=";")
        stats_dfs.append(stats_df.mean())
                
    all_stats = pd.DataFrame(stats_dfs, index=datasets)
    all_stats.to_csv(out_dir/'all_stats.csv', sep=";")
    print(all_stats.to_string())
                