# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'


import io
import os
import math
import glob
import time
import json
from collections import defaultdict, deque
import datetime
import numpy as np
from typing import List

try:
    from timm.utils import get_state_dict
except ImportError:
    get_state_dict = None

from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed_each_gpu(rank):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        builtin_print("rank is: ", rank, end=" ")
        now = datetime.datetime.now().time()
        builtin_print("[{}] ".format(now), end="")  # print with time stamp
        builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    if not args.enable_multi_print:
        setup_for_distributed(args.rank == 0)
    else:
        setup_for_distributed_each_gpu(args.rank)


def load_state_dict(
    model, state_dict, prefix="", ignore_missing="relative_position_index"
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


def save_model(
    output_dir,
    cfg,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    scheduler,
    loss_scaler,
    model_ema=None,
    exp_name=None,
):
    output_dir = Path(output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:  # distributed
        if exp_name is not None:
            checkpoint_paths = [
                output_dir / ("{}_checkpoint-{}.pth".format(exp_name, epoch_name))
            ]
        else:
            checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save_state_dict = model_without_ddp.state_dict()

            for key in list(to_save_state_dict.keys()):
                if key.startswith("teacher."):
                    to_save_state_dict.pop(key)

            to_save = {
                "model": to_save_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "cfg": cfg,
            }

            if model_ema is not None:
                to_save["model_ema"] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)

            all_checkpoints = glob.glob(os.path.join(output_dir, "*_checkpoint-*"))
            epochs = [
                int(Path(cp).name.split("-")[-1].split(".")[0])
                for cp in all_checkpoints
            ]
            epochs.sort()
            for e in epochs[:-2]:
                checkpoint_path = output_dir / (
                    "%s_checkpoint-%s.pth" % (cfg.info.exp_name, e)
                )
                checkpoint_path.unlink()

        return checkpoint_paths
    else:
        client_state = {"epoch": epoch}
        if model_ema is not None:
            client_state["model_ema"] = get_state_dict(model_ema)
        if exp_name is not None:
            model.save_checkpoint(
                save_dir=output_dir,
                tag="{}_checkpoint-{}".format(exp_name, epoch_name),
                client_state=client_state,
            )
        else:
            model.save_checkpoint(
                save_dir=output_dir,
                tag="checkpoint-%s" % epoch_name,
                client_state=client_state,
            )


def auto_load_model(
    cfg,
    model,
    device,
    auto_resume,
    model_without_ddp,
    optimizer,
    scheduler,
    loss_scaler,
    model_ema=None,
    resume=None,
):
    resume = cfg.reload.resume if resume is None else resume
    if loss_scaler is not None:
        # torch.amp
        if auto_resume and len(resume) == 0:
            ckpt_dir = cfg.dirs.ckp_dir
            all_checkpoints = glob.glob(os.path.join(ckpt_dir, "*_checkpoint-*.pth"))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                resume = os.path.join(ckpt_dir, "checkpoint-%d.pth" % latest_ckpt)
            print("Auto resume checkpoint: %s" % resume)

        if resume:
            if resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(resume, map_location="cpu")

            # handle ema model
            need_state_dict = model_without_ddp.state_dict()
            need_ema = False
            for key in need_state_dict.keys():
                if "teacher" in key:
                    need_ema = True
                    break

            checkpoint_model = checkpoint["model"]

            if need_ema:
                all_keys = list(checkpoint_model.keys())
                all_keys = [key for key in all_keys if key.startswith("encoder.")]
                for key in all_keys:
                    new_key = key.replace("encoder.", "teacher.")
                    checkpoint_model[new_key] = checkpoint_model[key]

            model_without_ddp.load_state_dict(checkpoint_model)
            print("Resume checkpoint %s" % resume)
            if "optimizer" in checkpoint and "epoch" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                cfg.start_epoch = checkpoint["epoch"] + 1
                if hasattr(cfg, "model_ema") and cfg.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
                if "scaler" in checkpoint:
                    loss_scaler.load_state_dict(checkpoint["scaler"])
                print("With optim & sched!")
            return True
        else:
            return False

    else:
        # deepspeed, only support '--auto_resume'.
        if auto_resume:
            import glob

            ckpt_dir = cfg.dirs.ckp_dir
            all_checkpoints = glob.glob(os.path.join(ckpt_dir, "*_checkpoint-*"))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                exp_name_file = Path(ckpt).name.split("_")[0]
                assert exp_name_file == cfg.info.exp_name
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                resume = os.path.join(
                    ckpt_dir, "%s_checkpoint-%d.pth" % (exp_name_file, latest_ckpt)
                )
                print("Auto resume checkpoint: %d" % latest_ckpt)

        if resume is not None:
            print("Resume checkpoint %s" % resume)
            # _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
            checkpoint = torch.load(resume, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            ) if optimizer is not None else None
            scheduler.load_state_dict(
                checkpoint["scheduler"]
            ) if scheduler is not None else None
            loss_scaler.load_state_dict(
                checkpoint["scaler"]
            ) if loss_scaler is not None else None
            cfg.reload.start_epoch = checkpoint["epoch"] + 1
            if model_ema is not None:
                if cfg.reload.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            return True
        else:
            return False


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
