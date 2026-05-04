"""Main entry point for training and testing models."""

import json
import math
import os
import sys
import time
from rich.console import Console
import yaml
from collections.abc import Iterable
from pathlib import Path

# Ensure project root is importable (for pipeline regression loader in test.data)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _preload_thread_env() -> None:
    """Read num_threads from Hydra config early and set env vars."""
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "run.yaml"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            if "num_threads" not in cfg:
                raise KeyError("num_threads is not defined in config")
            num_threads = int(cfg.get("num_threads"))

            for var in [
                "OMP_THREAD_LIMIT",
                "NUMBA_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]:
                os.environ[var] = str(num_threads)

            print(f"[INFO] Preloaded thread limit: {num_threads}")

    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        print(f"[WARN] Could not read num_threads from {cfg_path}: {exc}")
        raise


_preload_thread_env()

# shutil
from typing import Any, Union
import uuid

import hydra
import lightning as L
import numpy as np
import torch
import awkward as ak
import random

from threadpoolctl import threadpool_info
from joblib import parallel_backend

from lightning import Callback, Trainer, LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from hydra.core.hydra_config import HydraConfig

from picid.data.cache.file_lock import FileLock
from picid.data.data_objects import SplitViewPolicy
from picid.data.datasources.base.contracts import DatasourceProtocol

from picid.data.preprocessing.preprocessor import PreProcessor
from picid.exceptions import PreprocessingDatasourceError
from picid.evaluator.base import AbstractEvaluator
from picid.model.adapters.base import (
    AbstractFeedForwardWrapper,
    AbstractFitPredictWrapper,
    AbstractFeedForwardTrainingWrapper,
)
from picid.pipeline.base import (
    ConstantLossLightningModule,
    FitPredictWrapperLightningModule,
    TrainingLightningModule,
)
from picid.callbacks.model_checkpoint import ModelCheckpointWithConfig
from picid.transforms.base.multisource import InverseTransformMixin
from picid.transforms.base.transform_manager import ConfigTransformManager

from picid.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
    log_hyperparameters,
    display_targets,
    print_hydra_config_tree,
)
from picid.utils.logging_utils import get_hydra_override
from picid.utils.awkward_utils import get_ak_shape
from picid.utils.examine_model import get_model_summary

# register resolver before hydra.main
OmegaConf.register_new_resolver("flat", lambda s: s.replace("/", "+"))
OmegaConf.register_new_resolver("uuid", lambda kind="short": _uuid(kind))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("mod", lambda x, y: int(x) % int(y))
# integer division operator
OmegaConf.register_new_resolver("int_div", lambda a, b: int(a) // int(b))

OmegaConf.register_new_resolver("prod", lambda *args: np.prod(args))

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


# import threading, traceback

# _orig_start = threading.Thread.start


# def traced_start(self, *args, **kwargs):
#     print(f"\n[Thread started: {self.name}]")
#     traceback.print_stack(limit=5)
#     return _orig_start(self, *args, **kwargs)


# threading.Thread.start = traced_start


def safe_div(a, b):
    """
    Return ``a // b`` when ``a`` is evenly divisible by ``b``.

    If ``b`` is a non-string iterable, try each candidate divisor until one
    succeeds.

    Parameters
    ----------
    a : int
        Dividend.
    b : int or Iterable of int
        Divisor or iterable of candidate divisors (strings and bytes are not
        treated as iterables of divisors).

    Returns
    -------
    int
        Integer quotient.

    Raises
    ------
    ZeroDivisionError
        If a scalar divisor is zero.
    ValueError
        If division is not exact, no candidate succeeds, or the candidate list
        is empty.
    """
    # Treat lists/tuples as candidates, but avoid treating strings/bytes as iterables here.
    if isinstance(b, Iterable) and not isinstance(b, (str, bytes)):
        bs = list(b)
        if len(bs) == 0:
            raise ValueError("empty list of divisors")
        last_err = None
        for bi in bs:
            try:
                return safe_div(a, bi)
            except Exception as e:
                last_err = e
        raise ValueError(f"{a} is not divisible by any candidate in {bs}") from last_err

    if b == 0:
        raise ZeroDivisionError("division by zero")
    if a % b != 0:
        raise ValueError(f"{a} is not divisible by {b}")
    return a // b


OmegaConf.register_new_resolver("quot", lambda a, b: safe_div(a, b))


def _write_metrics_to_eval_details(
    cfg: DictConfig, stage: str, split: str, metrics: dict[str, Any]
) -> None:
    """
    Write metrics to ``eval_details/{stage}/{split}/metrics.json`` when enabled.

    Parameters
    ----------
    cfg : DictConfig
        Runtime configuration carrying output paths.
    stage : str
        Evaluation stage, such as ``"best_epoch"``.
    split : str
        Split name, such as ``"val"`` or ``"test"``.
    metrics : dict[str, Any]
        Metrics dictionary returned by Lightning.
    """
    if not metrics:
        return
    # Strip "val/" or "test/" prefix from keys
    clean = {k.split("/", 1)[1] if "/" in k else k: v for k, v in metrics.items()}
    p_dir = getattr(cfg.paths, "eval_details", None)
    if p_dir:
        out = Path(str(p_dir)) / stage / split / "metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, default=_json_default)
        log.info("Saved metrics to %s", out)


def _json_default(o: Any) -> Any:
    """
    Convert metric values into JSON-serializable scalars.

    Parameters
    ----------
    o : Any
        Value to serialize. Supports ``torch.Tensor``, NumPy scalar types, and
        Python ``float``. Non-finite floating values are written as strings.

    Returns
    -------
    Any
        JSON-serializable scalar or string.

    Raises
    ------
    TypeError
        If ``o`` is not a supported type.
    """
    if isinstance(o, torch.Tensor):
        v = o.detach().cpu().item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return str(v)
        return v
    if isinstance(o, (np.floating, np.integer)):
        v = float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return str(v)
        return v
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _log_preprocessing_datasource_error(exc: PreprocessingDatasourceError) -> None:
    """
    Render datasource preprocessing failures with explicit top-level context.

    Parameters
    ----------
    exc : PreprocessingDatasourceError
        Preprocessing exception raised while interacting with the datasource.

    Returns
    -------
    None
        The function logs additional context but does not suppress the error.
    """

    log.error("Datasource preprocessing failed.")
    if exc.stage is not None:
        log.error("Preprocessing datasource stage: %s", exc.stage)
    if exc.datasource_type is not None:
        log.error("Datasource class: %s", exc.datasource_type)
    if exc.datasource_name is not None:
        log.error("Datasource name: %r", exc.datasource_name)
    if exc.datasource_error_type is not None:
        log.error("Datasource error type: %s", exc.datasource_error_type)
    if exc.cause is not None:
        log.error("Original datasource error: %s", exc.cause)


# safe diff that returns a if a or b are strings
def diff_if_not_string(a, b):
    """
    Return ``a`` when either operand is a string; otherwise return ``a - b``.

    Parameters
    ----------
    a : Any
        Left-hand value.
    b : Any
        Right-hand value.

    Returns
    -------
    Any
        ``a`` if ``a`` or ``b`` is a string, else ``a - b``.
    """
    if isinstance(a, str) or isinstance(b, str):
        return a
    return a - b


OmegaConf.register_new_resolver("diff", lambda a, b: diff_if_not_string(a, b))


OmegaConf.register_new_resolver(
    "infer_data_dim", lambda key, dim: "not yet initialized"
)

OmegaConf.register_new_resolver(
    "infer_dataloader_length", lambda key: "not yet initialized"
)


def register_infer_dataloader_length_resolver(lengths: dict[str, int]) -> None:
    """
    Register the Hydra ``infer_dataloader_length`` resolver from length map.

    Parameters
    ----------
    lengths : dict[str, int]
        Mapping from dataloader keys to lengths.

    Returns
    -------
    None
        Resolver registration is performed in place.
    """

    def get_key(lengths, key):
        """
        Look up a dataloader length or raise with resolver context.

        Parameters
        ----------
        lengths : dict[str, int]
            Known lengths by key.
        key : str
            Resolver key.

        Returns
        -------
        int
            Length for ``key``.

        Raises
        ------
        KeyError
            If ``key`` is missing from ``lengths``.
        """
        if key not in lengths:
            raise KeyError(
                f"Resolver issue: key: '{key}' "
                f"not found in dataloader lengths dictionary ({lengths.keys()})."
            )
        return lengths[key]

    OmegaConf.register_new_resolver(
        "infer_dataloader_length",
        lambda key: get_key(lengths, key),
        replace=True,
    )


def register_data_dim_resolver(
    data: dict[str, Union[np.array, ak.Array, torch.Tensor]],
) -> None:
    """
    Register the Hydra ``infer_data_dim`` resolver from a data dictionary.

    Parameters
    ----------
    data : dict[str, Union[np.ndarray, ak.Array, torch.Tensor]]
        Mapping from resolver keys to array-like values.

    Returns
    -------
    None
        Resolver registration is performed in place.
    """

    def infer_data_dim(obj: Union[np.array, ak.Array, torch.Tensor], dim: int) -> int:
        """
        Infer tensor length along ``dim`` for supported array types.

        Parameters
        ----------
        obj : Union[list, np.ndarray, torch.Tensor, ak.Array]
            Array-like object or nested list of consistent arrays.
        dim : int
            Dimension index.

        Returns
        -------
        int
            Size along ``dim``.

        Raises
        ------
        ValueError
            If the type is unsupported or list elements disagree on shape.
        """
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
        """
        Return ``data[key]`` or raise with resolver context.

        Parameters
        ----------
        data : dict[str, Any]
            Resolver source mapping.
        key : str
            Requested key.

        Returns
        -------
        Any
            Value for ``key``.

        Raises
        ------
        KeyError
            If ``key`` is missing from ``data``.
        """
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
    """
    Build a UUID string for the ``uuid`` OmegaConf resolver.

    Parameters
    ----------
    kind : str
        ``\"hex\"`` for the full hex form, ``\"short\"`` for an 8-character prefix.

    Returns
    -------
    str
        UUID representation.

    Raises
    ------
    ValueError
        If ``kind`` is not recognized.
    """
    u = uuid.uuid4()
    if kind == "hex":
        return u.hex
    if kind == "short":
        return u.hex[:8]
    raise ValueError(f"Unknown uuid kind: {kind}")


def initialize_hydra() -> DictConfig:
    """
    Initialize Hydra when ``@hydra.main`` is not available, such as in tests.

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


def verify_thread_limits(expected: int):
    """
    Log configured thread-related environment variables and library pools.

    Parameters
    ----------
    expected : int
        Expected thread count from configuration (for log context only).

    Returns
    -------
    None
        Emits logs only.
    """
    log.info(f"[Thread sanity check] Expected threads: {expected}")

    # environment variables
    for var in [
        "OMP_THREAD_LIMIT",
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        val = os.environ.get(var)
        if val:
            log.info(f"{var:25s}: {val}")

    # native library threadpools
    try:
        for lib in threadpool_info():
            name = lib.get("internal_api", "unknown")
            num_threads = lib.get("num_threads")
            path = Path(lib.get("filepath", "")).name
            log.info(f"{name:12s} ({path:35s}) → {num_threads} threads")
    except Exception as e:
        log.error(f"Could not query threadpools: {e}")


def log_training_run_config(
    cfg: DictConfig,
    trainer: Trainer,
    model: L.LightningModule,
    logger: list[Logger],
):
    """
    Save the resolved config and push hyperparameters to experiment loggers.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration including ``paths``.
    trainer : Trainer
        Active Lightning trainer.
    model : L.LightningModule
        Model being trained.
    logger : list[Logger]
        Lightning loggers (for example WandB).

    Returns
    -------
    None
        Writes files and updates loggers in place.
    """
    # This is needed because wandb otherwise only tracks part of the config
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(cfg_container)
    OmegaConf.save(resolved_cfg, cfg.paths.output_dir + "/config_resolved.yaml")

    try:
        if HydraConfig.initialized():
            cfg_container["hydra_cfg"] = OmegaConf.to_container(
                HydraConfig.get(),
                resolve=False,
            )
    except Exception:
        pass  # e.g. when running from reproduce_from_run.py without Hydra context

    for lgr in logger:
        if isinstance(lgr, WandbLogger):
            exp_name = get_hydra_override("experiment")
            cfg_container["exp_name"] = exp_name
            lgr.experiment.config.update(cfg_container)

    if logger:
        model_summary = get_model_summary(model)
        log.info("Logging hyperparameters!")
        log_hyperparameters(
            {
                "cfg": cfg,
                "trainer": trainer,
                "model/params/total": model_summary[-1]["total_params"],
                "model/params/trainable": model_summary[-1]["trainable_params"],
            }
        )


def create_lightning_module(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    evaluators: list[AbstractEvaluator],
    loss: Any,
):
    """
    Instantiate the Lightning module that wraps the configured backbone.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``model``, ``optimization``, and ``datasource`` nodes.
    datamodule : LightningDataModule
        Data module (used for optional dataloader-dependent construction).
    evaluators : list of AbstractEvaluator
        Evaluators wired into the Lightning module.
    loss : Any
        Loss object from configuration.

    Returns
    -------
    L.LightningModule
        Training, constant-loss, or fit-predict Lightning module.

    Raises
    ------
    ValueError
        If the backbone wrapper type is not supported.
    AssertionError
        If no model could be built.
    """
    # we can only partially initialize the optimizer here because
    # we need the parameter list from the model later
    # as a consequence, the same applies for the scheduler
    # because we need the optimizer instance

    # Instantiate optimizer (partially)
    optimizer_factory = hydra.utils.instantiate(
        cfg.optimization.optimizer, _partial_=True
    )

    if (
        hasattr(cfg.optimization, "scheduler")
        and cfg.optimization.scheduler is not None
    ):
        # Instantiate scheduler (partially)
        scheduler_factory = hydra.utils.instantiate(
            cfg.optimization.scheduler,
            _partial_=True,
        )
        log.info(f"Scheduler: {cfg.optimization.scheduler._target_}")
    else:
        scheduler_factory = None

    log.info(f"Optimizer: {cfg.optimization.optimizer._target_}")
    # Check the LightningModules to see the full initialization of optimizer/scheduler

    model = None
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
                optimizer_factory=optimizer_factory,
                scheduler_factory=scheduler_factory,
            )
        elif isinstance(backbone_wrapper, AbstractFeedForwardTrainingWrapper):
            model = TrainingLightningModule(
                backbone=backbone_wrapper,
                loss=loss,
                evaluators=evaluators,
                optimizer_factory=optimizer_factory,
                scheduler_factory=scheduler_factory,
            )

        elif isinstance(backbone_wrapper, AbstractFitPredictWrapper):
            model = FitPredictWrapperLightningModule(
                backbone=backbone_wrapper,
                loss=loss,
                evaluators=evaluators,
                optimizer_factory=optimizer_factory,
                scheduler_factory=scheduler_factory,
                debug=cfg.get("debug", False),
            )
        else:
            raise ValueError(
                f"Unsupported backbone wrapper type: {type(backbone_wrapper)}"
            )

    else:
        if cfg.model.metadata.get("lightning_model", False):
            # Model is already a standard LightningModule and does not require a wrapper
            model = hydra.utils.instantiate(
                cfg.model,
                evaluators=evaluators,
                optimizer_factory=optimizer_factory,
                scheduler_factory=scheduler_factory,
            )

    assert model is not None, "Model could not be instantiated."
    return model


def rerun_best_model_checkpoint(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    device: torch.device,
    callbacks: list[Callback],
    evaluators: dict[str, AbstractEvaluator],
    loss: Any,
    logger: list[Logger],
) -> None:
    """
    Load the best ``ModelCheckpoint`` and re-run validation and test loops.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for trainer and model construction.
    datamodule : LightningDataModule
        Source of validation and test dataloaders.
    device : torch.device
        Device for the reloaded weights.
    callbacks : list[Callback]
        Callback list from training; must include ``ModelCheckpoint``.
    evaluators : dict[str, AbstractEvaluator]
        Evaluators forwarded to :func:`create_lightning_module`.
    loss : Any
        Loss for reconstructing the Lightning module.
    logger : list[Logger]
        Loggers that receive rerun metrics.

    Returns
    -------
    None
        Evaluation runs and metrics logging happen as side effects.

    Raises
    ------
    ValueError
        If no ``ModelCheckpoint`` callback is present.
    """
    for lgr in logger:
        if isinstance(lgr, WandbLogger):
            lgr.experiment.log({}, commit=True)

    checkpoint_model = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            log.info(
                f"Loading best model from checkpoint at {callback.best_model_path}"
            )
            model_path = Path(callback.best_model_path)
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

            checkpoint_model = create_lightning_module(
                cfg=cfg,
                datamodule=datamodule,
                evaluators=evaluators,
                loss=loss,
            )

            checkpoint_model.load_state_dict(ckpt["state_dict"], strict=True)
            checkpoint_model.to(device)
            break  # there is only one checkpoint callback

    if checkpoint_model is None:
        raise ValueError("No ModelCheckpoint callback found, cannot load best model.")

    # New trainer to log final metrics on validation set
    # Because wandb displays validation metrics from the final, not the best epoch.
    checkpoint_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        num_sanity_val_steps=0,
        enable_progress_bar=cfg.get("enable_progress_bar", True),
        logger=False,
    )

    log.info("Re-testing best model checkpoint on validation set!")
    val_loader = datamodule.val_dataloader()
    val_results = checkpoint_trainer.test(
        model=checkpoint_model, dataloaders=val_loader
    )
    if val_results:
        _write_metrics_to_eval_details(cfg, "best_epoch", "val", val_results[0])
        logged = {}
        for k, v in val_results[0].items():
            suffix = k.split("/", 1)[1] if "/" in k else k
            logged[f"val_best_rerun/{suffix}"] = v
        log.info(logged)
        for lgr in logger:
            if isinstance(lgr, WandbLogger):
                lgr.log_metrics(logged)
                lgr.experiment.log({}, commit=True)
                time.sleep(1.5)

    log.info("Re-testing best model checkpoint on test set!")
    test_loader = datamodule.test_dataloader()
    test_results = checkpoint_trainer.test(
        model=checkpoint_model, dataloaders=test_loader
    )
    if test_results:
        _write_metrics_to_eval_details(cfg, "best_epoch", "test", test_results[0])
        logged = {}
        for k, v in test_results[0].items():
            suffix = k.split("/", 1)[1] if "/" in k else k
            logged[f"test_best_rerun/{suffix}"] = v
        log.info(logged)
        for lgr in logger:
            if isinstance(lgr, WandbLogger):
                lgr.log_metrics(logged)
                lgr.experiment.log({}, commit=True)
                time.sleep(1.5)


from picid.model.tabfm_guard import check_tabfm_available as _check_tabfm_available


@task_wrapper
def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Train the model and optionally evaluate the best checkpoint on the test set.

    This method is wrapped in the optional ``@task_wrapper`` decorator, which
    controls failure handling. That makes it suitable for multiruns and crash
    reporting.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple with metrics and dict with all instantiated objects.
    """
    _check_tabfm_available(cfg)

    if cfg.get("num_threads", None) is not None:
        log.warning(f"Setting number of threads for all libraries to {cfg.num_threads}")
        verify_thread_limits(cfg.num_threads)
        parallel_backend("loky", n_jobs=cfg.num_threads)
        torch.set_num_threads(cfg.num_threads)
        torch.set_num_interop_threads(cfg.num_threads)
        log.info(f"Set number of threads to {cfg.num_threads}")
    else:
        log.info("Using default number of threads for all libraries")

    if not (fa_flag := cfg.get("flash_attention", True)):
        if not fa_flag:
            log.warning(
                "Flash attention is disabled. This may lead to increased memory usage."
            )
            #  If OOM errors occur, try to set these to False
            # because otherwise flash attention does not throw an error.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

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
    tree = print_hydra_config_tree(cfg)
    Console().print(tree)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiate and load dataset
    log.info(f"Instantiating loader <{cfg.datasource._target_}>")
    dataset_source: DatasourceProtocol = hydra.utils.instantiate(cfg.datasource)
    # Create manager
    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    # Get transform names
    log.info("Transform names:" + str(transforms_manager.get_transform_names()))

    # Get transforms that apply to features
    # feature_transforms = transforms_manager.get_transforms_by_apply_to("features")
    # log.info("Feature transforms:" + str(list(feature_transforms.keys())))

    split_mode = dataset_source.get_split_mode()
    log.info("Datasource split mode: %s", split_mode)

    preprocessor = PreProcessor(
        datasource=dataset_source, transforms=transforms_manager
    )

    # Get cache dirs and flags
    data_cache_path = (
        cfg.paths.cache_path
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    data_library_part_path = (
        Path(cfg.paths.root_dir) / "picid/data/datasources"
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    transform_library_part_path = (
        Path(cfg.paths.root_dir) / "picid/transforms"
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    cache_preprocessed_flag = cfg.cache.use_cache_after_transfroms
    use_file_lock = cfg.cache.get("use_preprocessing_file_lock", False)
    file_lock_path = cfg.cache.get(
        "preprocessing_file_lock_path", "/tmp/picid_preprocess.lock"
    )

    if use_file_lock:
        file_lock = FileLock(path=file_lock_path)
        log.info(
            f"Acquiring preprocessing file lock at {file_lock_path} to prevent concurrent preprocessing."
        )
        file_lock.acquire()
        log.info("Preprocessing file lock acquired.")

    try:
        # Run the preprocessing pipeline
        try:
            preprocessor.pipeline(
                data_cache_path=data_cache_path,
                data_library_part_path=data_library_part_path,
                transform_library_part_path=transform_library_part_path,
                cache_preprocessed=cache_preprocessed_flag,
            )
        except PreprocessingDatasourceError as exc:
            _log_preprocessing_datasource_error(exc)
            raise

        data_dict = preprocessor.get_processed_split_dict(
            view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
        )
        meta_data_dict = preprocessor.get_meta_data_dict()
        if cache_preprocessed_flag:
            transforms_manager = preprocessor.get_cached_transform_manager()
    finally:
        if use_file_lock:
            file_lock.release()
            log.info("Preprocessing file lock released.")

    assert set(data_dict.keys()) == {
        "train",
        "val",
        "test",
    }, f"Expected splits 'train', 'val', 'test', but got {set(data_dict.keys())}"

    # Check that the features are the same for all splits
    features_set = set(data_dict["train"].keys())
    for split in ["val", "test"]:
        assert set(data_dict[split].keys()) == features_set, (
            f"Mismatch in features for split {split}. "
            f"Expected: {features_set}, "
            f"Found: {set(data_dict[split].keys())}"
        )

    # Potentially we can delete this line hence offloading explicitly stating the input keys in the .yaml files
    input_data_task_keys = cfg.task_definition.model.data_requirements.input_tensors

    # 1. Create the datasets and store them in a dictionary
    datasets = {}
    for split in ["train", "val", "test"]:
        key = f"dataset_{split}"

        init_cfg = cfg.dataset
        if (oc := getattr(cfg.dataset, "_overrides", None)) is not None:
            if split in oc:
                overrides = cfg.dataset._overrides[split]
                init_cfg = OmegaConf.merge(cfg.dataset, overrides)
                log.info(f"Applied overrides for Dataset {split}: {overrides}")

        # TODO: Patch to allow association of subunit in the concat dataset
        meta_data_dict = meta_data_dict.copy()
        meta_data_dict["current_data_split"] = split

        datasets[key] = hydra.utils.instantiate(
            init_cfg,
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
    loader_lengths = {}
    for split, loader in zip(
        ["train", "val", "test"],
        [
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            datamodule.test_dataloader(),
        ],
    ):
        loader_lengths[split] = len(loader)
        batch = next(iter(loader))
        log.info(f"[1/{loader_lengths[split]}] Batch shapes for {split}:")

        nox_shapes_dict = {}

        for key, value in batch.items():
            # TODO: handle nested structures (attributedict, list, etc)
            if isinstance(value, torch.Tensor):
                log.info(f"  {key}: {value.shape}")
                nox_shapes_dict[key] = str(value.shape)
            else:
                log.info(f"  {key}: Non-tensor value of type {type(value)}")

        out_path = os.environ.get("NOX_RUN_DETAILS_JSON")
        if out_path is not None:
            # Save the shapes to a json file for nox to pick up
            out_path = Path(out_path)
            if out_path.parent.exists():
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(nox_shapes_dict, f)

    register_infer_dataloader_length_resolver(loader_lengths)

    evaluators = {}
    for s in ["train", "val", "test"]:
        inverse_transform_required = cfg.evaluator[s].get(
            "apply_inverse_scaling", False
        )
        inverse_transform_key = cfg.evaluator[s].get("inverse_transform_key", None)
        inverse_transform_which = cfg.evaluator[s].get(
            "inverse_transform_which", "last"
        )
        inverse_transform_name = cfg.evaluator[s].get("inverse_transform_name", None)

        inverse_transform = None
        if inverse_transform_required:
            # Prefer key-based lookup: discover inverter from registry
            if inverse_transform_key is not None:
                inverse_transform, chosen_name = (
                    transforms_manager.get_inverter_for_key_with_name(
                        inverse_transform_key, which=inverse_transform_which
                    )
                )
                if inverse_transform is None:
                    raise ValueError(
                        f"Evaluator for {s} requires inverse scaling and inverse_transform_key={inverse_transform_key!r} "
                        "was set, but no transform that assigns to that key implements InverseTransformMixin."
                    )
                if chosen_name:
                    log.info(
                        "Inverse transform for %s: using transform %r for key %r (which=%s).",
                        s,
                        chosen_name,
                        inverse_transform_key,
                        inverse_transform_which,
                    )
            elif inverse_transform_name is not None:
                inverse_transform = transforms_manager.get_transform(
                    inverse_transform_name
                ).transform_instance
                assert isinstance(inverse_transform, InverseTransformMixin), (
                    f"The transform {inverse_transform_name} does not implement "
                    "InverseTransformMixin but is selected for inverse_transform in evaluator."
                )
            else:
                raise ValueError(
                    f"Evaluator for {s} requires inverse scaling but neither "
                    "inverse_transform_key (e.g. 'rul') nor inverse_transform_name is provided."
                )

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

    model = create_lightning_module(
        cfg=cfg,
        datamodule=datamodule,
        evaluators=evaluators,
        loss=loss,
    )

    additional_trainer_args = {}
    # if isinstance(model, FitPredictWrapperLightningModule):
    #     additional_trainer_args["limit_val_batches"] = 0.0

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        **additional_trainer_args,
        enable_progress_bar=cfg.get("enable_progress_bar", True),
    )

    log_training_run_config(
        cfg=cfg,
        trainer=trainer,
        model=model,
        logger=logger,
    )

    if cfg.task_definition.get("requires_training", False):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

    if cfg.trainer.max_epochs > 1:
        # Intention is to reload all deep-learning trained models from checkpoint
        test_results = trainer.test(ckpt_path="best", datamodule=datamodule)
    else:
        # If max epochs is 1, there is no checkpoint to load from
        # So we just test the model as is
        log.info("Starting testing without training (used for pre-trained models)!")
        test_results = trainer.test(model=model, datamodule=datamodule)

    if test_results:
        _write_metrics_to_eval_details(cfg, "best_epoch", "test", test_results[0])

    # We want to avoid running this for pfn models where there is no training
    # and thus no checkpoint and rerunning pfn model would take significant time.
    if cfg.task_definition.get("requires_training", False) and (
        cfg.trainer.max_epochs > 1
    ):
        time.sleep(1.5)
        rerun_best_model_checkpoint(
            cfg=cfg,
            datamodule=datamodule,
            device=trainer.strategy.root_device,
            callbacks=callbacks,
            evaluators=evaluators,
            loss=loss,
            logger=logger,
        )

    for lgr in logger:
        if isinstance(lgr, WandbLogger):
            lgr.experiment.finish()

    return None, None  # metric_dict, object_dict


@hydra.main(version_base="1.3", config_path=str(_PROJECT_ROOT / "configs"), config_name="run.yaml")
def main(cfg: DictConfig) -> float | None:
    """
    Main entry point for training.

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
