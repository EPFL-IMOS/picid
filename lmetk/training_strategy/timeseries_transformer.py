import shutil
import time
import torch
import os
import json
import datetime
import math
import sys
import hydra


import pandas as pd
from pathlib import Path
from typing import Iterable
from omegaconf import OmegaConf
import numpy as np
from einops import rearrange


import utils.distutil as utils
from utils.distutil import (
    NativeScalerWithGradNormCount as NativeScaler,
    auto_load_model,
)


class TrainingBase(object):
    def __init__(self) -> None:
        super().__init__()


class TrainSTF(TrainingBase):
    def __init__(self, cfg, objects) -> None:
        super().__init__()
        self.objects = objects
        self.distributed = cfg.glob.distributed
        self.cfg = cfg

    def call_scheduler(self, scheduler, metric=None, is_end_epoch=False):
        if metric is not None:
            scheduler.step(metric, is_end_epoch=is_end_epoch)
        else:
            scheduler.step()

    def train(self):
        # ingredients
        cfg = self.cfg
        data_loader_train = self.objects["train_data_loader"]
        data_loader_val = self.objects["val_data_loader"]

        device = self.objects["device"]
        model = self.objects["model"]
        model_without_ddp = self.objects["model_without_ddp"]

        optimizer = self.objects["optimizer"]
        scheduler = self.objects["scheduler"]
        lr_schedule_values = self.objects["lr_schedule_values"]
        wd_schedule_values = self.objects["wd_schedule_values"]
        best_metrics_dict = self.objects["best_metrics_dict"]
        loss_scaler = NativeScaler()

        start_time = time.time()
        start_epoch = 0 if cfg.reload.resume is False else cfg.reload.start_epoch

        self.call_scheduler(scheduler)

        for epoch in range(start_epoch, cfg.trainer.epochs):
            if cfg.glob.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            if cfg.lightning.use_lightning_dataloader is False:
                start_steps = epoch * cfg.trainer.num_training_steps_per_epoch
            else:
                start_steps = epoch * len(data_loader_train)

            train_stats, global_step = self.train_one_epoch(
                cfg=cfg,
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                lr_scheduler=scheduler,
                start_steps=start_steps,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
            )

            val_stats = self.validate(
                cfg=cfg,
                model=model,
                vali_loader=data_loader_val,
                device=device,
                epoch=epoch,
                log_str="val",
            )

            if scheduler is not None:
                self.call_scheduler(
                    scheduler, metric=val_stats["loss"], is_end_epoch=True
                )

            if val_stats["loss"] < best_metrics_dict["val_loss"][0]:
                best_metrics_dict["val_loss"] = (
                    val_stats["loss"].item(),
                    epoch,
                    global_step,
                )
                save_best_val = True
            else:
                save_best_val = False

            if cfg.dirs.ckp_dir:
                if (
                    (epoch + 1) % cfg.reload.save_ckpt_freq == 0
                    or epoch + 1 == cfg.trainer.epochs
                    or save_best_val
                ):
                    checkpoint_paths = utils.save_model(
                        output_dir=cfg.dirs.ckp_dir,
                        cfg=cfg,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        exp_name=cfg.info.exp_name,
                    )

                    if save_best_val and utils.is_main_process():
                        dir = Path(cfg.dirs.best_ckp_dir)
                        for f in dir.iterdir():
                            if f.name.startswith("best_val_"):
                                f.unlink()
                        shutil.copy2(checkpoint_paths[0], dir / f"best_val_{epoch}.pth")
                        OmegaConf.save(best_metrics_dict, dir / "best_metrics.yaml")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }

            if cfg.dirs.log_dir and utils.is_main_process():
                with open(
                    os.path.join(cfg.dirs.log_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def test(self, best_metrics_dict):
        # ingredients
        cfg = self.cfg
        if utils.is_main_process():
            dir = Path(cfg.dirs.best_ckp_dir)
            for f in dir.iterdir():
                if f.name.startswith("best_val_"):
                    f.name.split("_")[-1].split(".")[0]

            data_loader_test = self.objects["test_data_loader"]
            device = self.objects["device"]
            model = self.objects["model"]

            auto_load_model(
                cfg=cfg,
                auto_resume=False,
                model=model,
                model_without_ddp=None,
                optimizer=None,
                scheduler=None,
                loss_scaler=None,
                model_ema=None,
                device=device,
                resume=Path(cfg.dirs.best_ckp_dir)
                / f"best_val_{best_metrics_dict['val_loss'][1]}.pth",
            )

            model.to(device)

            test_stats = self.validate(
                cfg=cfg,
                model=model,
                vali_loader=data_loader_test,
                device=device,
                epoch=cfg.reload.start_epoch,
                log_str="test",
            )

            print("Test results: ", test_stats)

    def prepare_batch(self, batch, device, cfg):
        if cfg.lightning.use_lightning_forward is False:
            raise NotImplementedError("Not implemented yet")
        else:
            batch_device = [b.float().to(device, non_blocking=True) for b in batch]
            return batch_device

    def model_forward(self, model, batch_device, cfg, train):
        if cfg.lightning.use_lightning_forward is True:

            def process_one_batch(batch):
                outs = model.step(batch, train=train)
                return outs

            outs = process_one_batch(batch_device)
            return outs
        else:
            raise NotImplementedError("Not implemented yet")

    def validate(self, model, cfg, vali_loader, epoch, device, log_str="val"):
        print_freq = 20
        header = "Epoch: [{}] -{}-".format(epoch, log_str)
        metric_logger = utils.MetricLogger(delimiter="  ")

        model.eval()
        total_loss = []
        mean_logs = {}

        with torch.no_grad():
            for _, batch in enumerate(
                metric_logger.log_every(vali_loader, print_freq, header)
            ):
                batch_device = self.prepare_batch(batch, device, cfg)
                outs = self.model_forward(model, batch_device, cfg, train=False)
                loss_value = outs["loss"].mean().item()
                total_loss.append(loss_value)

                outs_log = {
                    k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                    for k, v in outs.items()
                }

                for k, v in outs_log.items():
                    if k not in mean_logs:
                        mean_logs[k] = []
                    mean_logs[k].append(v)

        mean_logs = {
            k if k != "loss" else "mean_loss": np.mean(v) for k, v in mean_logs.items()
        }
        total_loss = np.average(total_loss)
        model.train()
        stats = dict(loss=total_loss, **mean_logs)
        return stats

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        lr_scheduler=None,
        start_steps=None,
        lr_schedule_values=None,
        wd_schedule_values=None,
        cfg=None,
    ):
        print_freq = 150
        header = "Epoch: [{}]".format(epoch)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )

        model.train(True)
        optimizer.zero_grad()
        accum_iter = cfg.trainer.gradient_accumulation.accum_iter

        if start_steps is None:
            start_steps = epoch * len(data_loader)

        for step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            it = start_steps + step

            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = (
                            lr_schedule_values[it] * param_group["lr_scale"]
                        )
                    if (
                        wd_schedule_values is not None
                        and param_group["weight_decay"] > 0
                    ):
                        param_group["weight_decay"] = wd_schedule_values[it]

            batch_device = self.prepare_batch(batch, device, cfg)
            outs = self.model_forward(model, batch_device, cfg, train=True)

            loss = outs["loss"].mean()
            loss_value = outs["loss"].mean().item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss.backward()
            optimizer.step()

            if (step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)

            min_lr = 10.0
            max_lr = 0.0
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)

            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)

            if lr_scheduler is not None:
                self.call_scheduler(scheduler=lr_scheduler)

        metric_logger.synchronize_between_processes()
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(now_time, "Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, it


class TrainLSTM(TrainSTF):
    def __init__(self, cfg, objects) -> None:
        super(TrainLSTM, self).__init__(cfg, objects)

    def call_scheduler(self, scheduler, metric=None, is_end_epoch=False):
        if metric is not None:
            scheduler.step(metric)
