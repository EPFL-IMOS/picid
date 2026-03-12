import os
from logging import Logger
from pathlib import Path
from typing import List

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from lightning import Callback, LightningDataModule, Trainer
from omegaconf import OmegaConf, DictConfig, open_dict
from torch import nn

from picid.config import project_config
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader
from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.interface.model_wrapper import CustomModelWrapper
from picid.interface.schemas.evaluators import DefaultEvaluatorConfig
from picid.interface.schemas.loggers import BaseLogger
from picid.interface.schemas.model import AbsModelConfig
from picid.interface.schemas.task_definition import BaseTaskDefinition
from picid.interface.utils import ProcessedDatasource, InterfacePreProcessor, register_data_dim_resolver, \
    create_lightning_module
from picid.transforms.base.data_transform import DataTransform


class EasyInterface:
    """
        A high-level interface for managing machine learning experiments,
        data source discovery, configuration composition, and model training.
    """

    def __init__(self):
        """
            Initializes the EasyInterface, sets up internal configuration tracking,
            and populates the list of available data sources from the config path.
        """
        self._model_cfg = None
        self._datasource_cfg = None
        self._experiment_cfg = None
        self._transforms_cfg = None
        self._current_datasource = None

        self._config_path = project_config.config_path

        self.available_datasources = self.get_available_resource('datasource')

    def get_datasource(self, v: str):
        """
                Loads and instantiates, using Hydra, a data source by its name.

                Args:
                    v (str): The name or path identifier of the source.

                Returns:
                    AbstractDataSourceLoader: An instance of the requested data source, instantiated via Hydra.
        """
        datasource = self.load_resource(v, 'datasource')
        dataset_source = hydra.utils.instantiate(datasource)

        self._current_datasource = dataset_source
        return dataset_source

    def process_datasource(self, datasource: str | AbstractDataSourceLoader,
                           transforms: list[DataTransform]):
        """
                Applies a sequence of data transformations to a data source using
                a InterfacePreProcessor.

                Args:
                    datasource (str | AbstractDataSourceLoader): The datasource to process.
                    transforms (list[DataTransform]): A list of transformation objects
                        to apply to the data.

                Returns:
                    ProcessedDatasource: The result of the preprocessing pipeline,
                        typically containing features and targets.
        """

        # preprocessor_mode = "per_unit"
        #
        # if isinstance(datasource, MultiSourceLoader):
        #     preprocessor_mode = "cross_unit"

        preprocessor = InterfacePreProcessor(datasource,
                                             transforms,
                                             preprocessor_mode=None,
                                             cache_path=project_config.cache_path)

        return preprocessor.pipeline()

    @staticmethod
    def load_resource(v: str | Path | dict = None,
                      resource_type: str = None):
        """
            Resolves and loads a configuration resource (datasource, model, etc.)
            from a dictionary, string name, or file path.

            Args:
                v (str | Path | dict): The resource definition or reference.
                resource_type (str): The category of resource (e.g., 'datasource').

            Returns:
                DictConfig: The loaded OmegaConf configuration object.

            Raises:
                AssertionError: If the resource name is not found in the
                    available resources for that type.
        """

        if isinstance(v, dict):
            cfg = OmegaConf.create(v)
        elif isinstance(v, (str, Path)):
            v = Path(v)
            root = project_config.config_path / resource_type

            if v.is_relative_to(root):
                v = v.relative_to(root)

            max_level = None
            if resource_type == 'datasource':
                max_level = 1

            resources = EasyInterface.get_available_resource(str(resource_type),
                                                            max_level=max_level)

            v = str(v)

            assert v in resources, (f'Given path {v.replace('.yaml', '')} is not present in the resource type {resource_type}. '
                                         f'Available sources are {resources}.')

            if resource_type == 'datasource':
                v = v + '.yaml'

            path = root / v
            cfg = OmegaConf.load(path)
        else:
            assert False, f'Setter input must be str, Path or dict ({type(v)} given)'

        if resource_type == 'datasource':
            cfg['cache_dir'] = project_config.cache_dir
            cfg['data_path'] = project_config.data_dir / cfg['_target_'].split('.')[-1]

        return cfg

    @staticmethod
    def get_available_resource(resource_type: str, max_level: int = None):
        """
            Scans the internal configuration directory to list all available resource files
            for a specific type.

            Args:
                resource_type (str): The subdirectory to scan (e.g., 'model', 'datasource').
                max_level (int, optional): The maximum directory depth to search.

            Returns:
                list[str]: A list of relative paths (identifiers) for available resources.
        """

        if max_level is None:
            max_level = np.inf

        path = Path(project_config.config_path)
        path = path / resource_type

        all_files = []

        level = 0

        for root, dirs, files in os.walk(path):
            level += 1
            for file in files:
                all_files.append(str((Path(root) / file).relative_to(path)))

            if level > max_level:
                break

        if resource_type == 'datasource':
            all_files = [f.replace('.yaml', '') for f in all_files]

        return all_files

    def compose_config_file(self,
                            *,
                            task_definition: BaseTaskDefinition,
                            model: AbsModelConfig | CustomModelWrapper,
                            datasource: str | ProcessedDatasource,
                            evaluators,
                            callbacks: dict | Callback | list[Callback] | None = None,
                            transforms: str | dict = None,
                            overrides: List[str] = None
                            ) -> DictConfig:
        """
            Assembles a full Hydra experiment configuration by merging model,
            task, data, and callback configurations.

            Args:
                task_definition (BaseTaskDefinition): Definition of the learning task.
                model (AbsModelConfig): Configuration for the architecture.
                datasource (str | ProcessedDatasource): Data identifier or processed object.
                callbacks: Configuration for training callbacks.
                training_model: Optional training-specific overrides.
                transforms: Data transformation configurations.
                overrides (List[str]): Manual Hydra overrides (e.g., ['+params.lr=0.01']).

            Returns:
                DictConfig: A unified configuration object ready for Hydra instantiation.
        """

        training_model_cfg = None
        model_cfg = None

        if not isinstance(model, CustomModelWrapper):
            model_config_path = (project_config.config_path /
                                 'model' /
                                 (model.config_name + '.yaml'))
            model_cfg = OmegaConf.load(model_config_path)
            model_cfg.update(model.model_dump(by_alias=True))

            training_model_path = (project_config.config_path /
                                   'model_configs'/
                                   task_definition.config_name /
                                   (model.config_name + '.yaml'))
            training_model_cfg = OmegaConf.load(training_model_path)

        overrides = overrides if overrides is not None else []
        #  TO CHECK POSSIBLE, but almost certain, ERRORS AND INCOMPATIBILITIES
        local_instance = GlobalHydra.instance()
        GlobalHydra.instance().clear()

        cs = ConfigStore.instance()

        if isinstance(datasource, ProcessedDatasource):
            overrides.append(f'+datasource.task_mode={datasource.task_mode}')
        elif isinstance(datasource, str):
            datasource = self.load_resource(datasource, 'datasource')
            cs.store(group="datasource",
                     name="datasource_cfg",
                     node=datasource)
            overrides.append('datasource=datasource_cfg')
        else:
            raise ValueError('datasource must be a string (see available_datasources attribute) or '
                             'a ProcessedDatasource (using the process_datasource function).')

        if model_cfg is not None:
            for i, d in enumerate(training_model_cfg.defaults):
                if 'override /model' in d.keys():
                    del training_model_cfg.defaults[i]
                    break

            cs.store(group="model", name="model_cfg",
                     node=model_cfg)
            overrides.append('model=model_cfg')

        cs.store(group="task_definition",
                 name="task_definition_cfg",
                 node=task_definition.model_dump(by_alias=True))
        overrides.append('task_definition=task_definition_cfg')

        if training_model_cfg is not None:
            cs.store(group="model_configs", name="training_model_cfg",
                     node=training_model_cfg, package="_global_")
            overrides.append('+model_configs=training_model_cfg')

        if transforms is not None:
            cs.store(group="transforms", name="transforms_cfg",
                     node=transforms)
            overrides.append('+transforms=transforms_cfg')

        if callbacks is None or isinstance(callbacks, dict):
            if isinstance(callbacks, dict):
                callbacks_cfg = OmegaConf.load(str(callbacks))
            else:
                callbacks_cfg = OmegaConf.load(project_config.config_path / 'callbacks' / 'default.yaml')

            cs.store(group="callbacks",
                     name="callbacks_cfg",
                     node=callbacks_cfg)
            overrides.append('callbacks=callbacks_cfg')

        if evaluators is None or isinstance(evaluators, dict):
            if isinstance(evaluators, dict):
                evaluators_cfg = {s: v.model_dump(by_alias=True) for s, v in evaluators.items()}
            else:
                evaluators_cfg = OmegaConf.load(project_config.config_path / 'evaluator' / 'rul.yaml')

            cs.store(group="evaluator",
                     name="evaluator_cfg",
                     node=evaluators_cfg)
            overrides.append('evaluator=evaluator_cfg')

        cs.store(group="paths",
                 name="paths_cfg",
                 node=project_config.model_dump())
        overrides.append('paths=paths_cfg')

        with hydra.initialize_config_dir(config_dir=str(project_config.config_path.absolute()),
                                         job_name="interface", version_base="1.3"):

            cfg = hydra.compose(config_name='run',
                                overrides=overrides,
                                return_hydra_config=True)

        GlobalHydra.set_instance(local_instance)

        return cfg

    def train(self,
              output_dir: object = None,
              *,
              task_definition: BaseTaskDefinition,
              model: AbsModelConfig | CustomModelWrapper,
              datasource: str | None | ProcessedDatasource = None,
              callbacks: dict | Callback | list[Callback] | None = None,
              loggers: list[Logger | BaseLogger] = None,
              evaluators : dict[str, DefaultEvaluatorConfig] = None,
              transforms: list[DataTransform] = [],
              overrides: List[str] = None,
              custom_backbone: nn.Module = None,
              enable_progress_bar: bool = True) -> None:
        """
            The main entry point for running an experiment. It handles data loading,
            preprocessing, dataset instantiation, and PyTorch Lightning trainer execution.

            Args:
                output_dir (object): Directory where logs and weights are saved.
                task_definition (BaseTaskDefinition): Definition of the task and its metrics.
                model (AbsModelConfig): The model configuration to use.
                datasource: The source data identifier.
                callbacks: Training callbacks.
                loggers (list): List of loggers for tracking metrics (e.g., TensorBoard, WandB).
                transforms (list): Data transformations to apply before training.
                overrides (list): Hydra configuration overrides.
                enable_progress_bar (bool): Whether to show the Lightning progress bar.

            Returns:
                None: This function initiates the `trainer.fit()` process.
        """

        cfg = self.compose_config_file(datasource=datasource,
                                       task_definition=task_definition,
                                       evaluators=evaluators,
                                       callbacks=callbacks,
                                       overrides=overrides,
                                       model=model)

        if isinstance(datasource, ProcessedDatasource):
            data_dict = datasource.data_dict
            meta_data_dict = datasource.meta_data_dict
        else:
            datasource = self.load_resource(datasource, 'datasource')

            dataset_source = hydra.utils.instantiate(datasource)
            process_datasource = self.process_datasource(dataset_source, transforms)

            data_dict = process_datasource.data_dict
            meta_data_dict = process_datasource.meta_data_dict

        register_data_dim_resolver(data=data_dict["train"])

        features_set = set(data_dict["train"].keys())
        for split in ["val", "test"]:
            assert set(data_dict[split].keys()) == features_set, (
                f"Mismatch in features for split {split}. "
                f"Expected: {features_set}, "
                f"Found: {set(data_dict[split].keys())}")

        input_data_task_keys = cfg.task_definition.model.data_requirements.input_tensors

        datasets = {}
        for split in ["train", "val", "test"]:
            key = f"dataset_{split}"

            init_cfg = cfg.dataset
            if (oc := getattr(cfg.dataset, "_overrides", None)) is not None:
                if split in oc:
                    overrides = cfg.dataset._overrides[split]
                    init_cfg = OmegaConf.merge(cfg.dataset, overrides)

            meta_data_dict = meta_data_dict.copy()
            meta_data_dict["current_data_split"] = split

            datasets[key] = hydra.utils.instantiate(
                init_cfg,
                data_dict={key: data_dict[split][key] for key in input_data_task_keys},
                meta_data_dict=meta_data_dict)

        dataset_train = datasets["dataset_train"]
        dataset_val = datasets["dataset_val"]
        dataset_test = datasets["dataset_test"]

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

        evaluators = {}
        for s in ["train", "val", "test"]:
            evaluator = hydra.utils.instantiate(
                cfg.evaluator[s],
                inverse_transform=None,
                remote_logger=None,
            )
            evaluators[s] = evaluator

        loss = hydra.utils.instantiate(cfg.loss)

        if 'callbacks' in cfg:
            callbacks: list[Callback] = list(hydra.utils.instantiate(cfg["callbacks"]).values() if callbacks is not None else [])

        log_functions = []

        if loggers is None:
            loggers = []

        for l in loggers:
            if isinstance(l, Logger):
                log_functions.append(l)
            elif isinstance(l, BaseLogger):
                d = l.model_dump(by_alias=True)
                d['save_dir'] = project_config.save_dir
                l = hydra.utils.instantiate(d)
                log_functions.append(l)

        if 'config_name' in cfg.model:
            with open_dict(cfg):
                del cfg.model['config_name']

        model = create_lightning_module(
            backbone=model if isinstance(model, CustomModelWrapper) else None,
            cfg=cfg,
            datamodule=datamodule,
            evaluators=evaluators,
            loss=loss,
        )

        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=log_functions,
            num_sanity_val_steps=0,
            # **additional_trainer_args,
            enable_progress_bar=enable_progress_bar,
        )

        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

