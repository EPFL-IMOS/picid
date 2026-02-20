"""Detailed building blocks for a GraphBench-style facade over picid.run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import hydra
import lightning as L
from hydra import (
    initialize as hydra_initialize,
    initialize_config_dir as hydra_init_config_dir,
    compose as hydra_compose,
)
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from picid.run import create_lightning_module, rerun_best_model_checkpoint
from picid.data.preprocessing.preprocessor import PreProcessor
from picid.transforms.base.transform_manager import ConfigTransformManager
from picid.data.datasources.base.single_source_loader import SingleSourceLoader
from picid.data.datasources.base.phmd_loader import PHMDMultiSourceLoader
from picid.data.datasources.kubric_movie.loader import MOViLoader
from picid.data.datasources.nb14.loader import NB14Loader
from picid.data.datasources.unibo.loader import UNIBO21Loader
from picid.data.datasources.MZVAV import MZVAVLoader
from picid.utils import instantiate_callbacks, instantiate_loggers
from picid.data.cache.file_lock import FileLock

# Classes considered single-source to decide preprocessing mode.
SINGLE_SOURCE_TYPES: tuple[type, ...] = (
    SingleSourceLoader,
    PHMDMultiSourceLoader,
    MOViLoader,
    NB14Loader,
    UNIBO21Loader,
    MZVAVLoader,
)


class Experiment:
    """Compose Hydra config with overrides while hiding Hydra from consumers."""

    def __init__(self, overrides: list[str] | None = None):
        cfg_dir = Path(__file__).resolve().parent.parent / "configs"
        with hydra_init_config_dir(
            version_base="1.3", config_dir=str(cfg_dir), job_name="simplified"
        ):
            self.cfg = hydra_compose(
                config_name="run.yaml",
                overrides=overrides or [],
                return_hydra_config=True,
            )
        HydraConfig.instance().set_config(self.cfg)


class Loader:
    """Sets up datasource -> preprocessing -> datasets -> datamodule."""

    def __init__(self, cfg, transforms: "Transforms"):
        self.cfg = cfg
        self.transforms = transforms

    def load(self):
        # Instantiate datasource and transform manager from config.
        source = hydra.utils.instantiate(self.cfg.datasource)
        transform_manager = self.transforms.manager

        # Choose per-unit vs cross-unit preprocessing.
        mode = "per_unit" if isinstance(source, SINGLE_SOURCE_TYPES) else "cross_unit"
        preprocessor = PreProcessor(mode, source, transform_manager)

        # Cache settings (mirrors run.py).
        use_cache = (
            self.cfg.cache.use_cache_after_loading
            or self.cfg.cache.use_cache_after_transfroms
        )
        data_cache_path = self.cfg.paths.cache_path if use_cache else None
        data_library_part_path = (
            Path(self.cfg.paths.root_dir) / "picid/data/datasources"
            if use_cache
            else None
        )
        transform_library_part_path = (
            Path(self.cfg.paths.root_dir) / "picid/transforms" if use_cache else None
        )
        cache_preprocessed_flag = self.cfg.cache.use_cache_after_transfroms
        use_file_lock = self.cfg.cache.get("use_preprocessing_file_lock", False)
        file_lock_path = self.cfg.cache.get(
            "preprocessing_file_lock_path", "/tmp/picid_preprocess.lock"
        )

        file_lock = FileLock(path=file_lock_path) if use_file_lock else None
        if file_lock:
            file_lock.acquire()

        try:
            preprocessor.pipeline(
                data_cache_path=data_cache_path,
                data_library_part_path=data_library_part_path,
                transform_library_part_path=transform_library_part_path,
                cache_preprocessed=cache_preprocessed_flag,
            )
            if cache_preprocessed_flag:
                transform_manager = preprocessor.get_cached_transform_manager()
                self.transforms.manager = transform_manager
        finally:
            if file_lock:
                file_lock.release()

        # Collect processed splits.
        meta_data_dict = preprocessor.get_meta_data_dict()
        data = preprocessor.get_processed_data_dict(return_splits_on_first_level=True)
        input_data_task_keys = (
            self.cfg.task_definition.model.data_requirements.input_tensors
        )

        # 1. Create the datasets and store them in a dictionary
        cfg = self.cfg

        datasets = {}
        for split in ["train", "val", "test"]:
            key = f"dataset_{split}"

            init_cfg = cfg.dataset
            if (oc := getattr(cfg.dataset, "_overrides", None)) is not None:
                if split in oc:
                    overrides = cfg.dataset._overrides[split]
                    init_cfg = OmegaConf.merge(cfg.dataset, overrides)

            # TODO: Patch to allow association of subunit in the concat dataset
            meta_data_dict = meta_data_dict.copy()
            meta_data_dict["current_data_split"] = split

            datasets[key] = hydra.utils.instantiate(
                init_cfg,
                data_dict={key: data[split][key] for key in input_data_task_keys},
                meta_data_dict=meta_data_dict,
            )

        # Wrap into a LightningDataModule.
        datamodule = hydra.utils.instantiate(
            self.cfg.datamodule,
            dataset_train=datasets["dataset_train"],
            dataset_val=datasets["dataset_val"],
            dataset_test=datasets["dataset_test"],
        )
        return data, datamodule


class Transforms:
    """Thin convenience wrapper around ConfigTransformManager."""

    def __init__(self, manager: ConfigTransformManager):
        self.manager = manager

    @classmethod
    def build(cls, cfg):
        """Construct transform manager from config."""
        return cls(ConfigTransformManager(cfg.transforms))

    def names(self) -> Iterable[str]:
        return self.manager.get_transform_names()

    def apply(self, name: str, batch: dict[str, Any]):
        transform = self.manager.get_transform(name)
        return transform(batch)


class Evaluator:
    """Construct evaluators for train/val/test splits."""

    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, loggers=None):
        remote_logger = next(
            (lg for lg in (loggers or []) if lg.__class__.__name__ == "WandbLogger"),
            None,
        )
        return {
            split: hydra.utils.instantiate(
                self.cfg.evaluator[split],
                inverse_transform=None,
                remote_logger=remote_logger,
            )
            for split in ("train", "val", "test")
        }


class Model:
    """Create and train/test the Lightning model."""

    def __init__(self, cfg):
        self.cfg = cfg

    def optimize(self, datamodule, evaluators):
        loss = hydra.utils.instantiate(self.cfg.loss)
        callbacks = instantiate_callbacks(self.cfg.get("callbacks"))
        loggers = instantiate_loggers(self.cfg.get("logger"))
        model = create_lightning_module(self.cfg, datamodule, evaluators, loss)

        trainer: L.Trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            callbacks=callbacks,
            logger=loggers,
            num_sanity_val_steps=0,
            enable_progress_bar=self.cfg.get("enable_progress_bar", True),
        )

        if self.cfg.task_definition.get("requires_training", False):
            trainer.fit(model=model, datamodule=datamodule)

        trainer.test(
            model=None if self.cfg.trainer.max_epochs > 1 else model,
            ckpt_path="best" if self.cfg.trainer.max_epochs > 1 else None,
            datamodule=datamodule,
        )

        if self.cfg.task_definition.get("requires_training", False) and (
            self.cfg.trainer.max_epochs > 1
        ):
            rerun_best_model_checkpoint(
                self.cfg,
                datamodule,
                trainer.strategy.root_device,
                callbacks,
                evaluators,
                loss,
                loggers,
            )

        return {}, {"trainer": trainer, "model": model}
