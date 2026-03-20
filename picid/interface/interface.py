import logging
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Iterable

import hydra
import numpy as np

from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.core.hydra_config import HydraConfig
import yaml

from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks.callback import Callback as PytorchCallback
from pytorch_lightning.callbacks.callback import Callback as DeprecatedPytorchCallback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger

# from threadpoolctl import threadpool_info
# from joblib import parallel_backend

import torch
import random

from picid.config import project_config
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader
from picid.interface.model.custom_model import CustomModelTrainer
from picid.interface.schemas.evaluators import DefaultEvaluatorConfig
from picid.interface.schemas.loggers import BaseLogger
from picid.interface.schemas.model import AbsModelConfig
from picid.interface.schemas.task_definition import BaseTaskDefinition
from picid.interface.utils import ProcessedDatasource, InterfacePreProcessor, register_data_dim_resolver, \
    create_lightning_module
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base.transform_manager import ConfigTransformManager
from picid.callbacks.model_checkpoint import ModelCheckpointWithConfig
from picid.transforms.base.multisource import InverseTransformMixin
# from picid.utils import display_targets

def verify_thread_limits(expected: int):
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

def register_infer_dataloader_length_resolver(lengths: dict[str, int]) -> None:
    def get_key(lengths, key):
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

def safe_div(a, b):
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


# safe diff that returns a if a or b are strings
def diff_if_not_string(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a
    return a - b

def _uuid(kind: str) -> str:
    u = uuid.uuid4()
    if kind == "hex":
        return u.hex
    if kind == "short":
        return u.hex[:8]
    raise ValueError(f"Unknown uuid kind: {kind}")


OmegaConf.register_new_resolver("quot", lambda a, b: safe_div(a, b))
OmegaConf.register_new_resolver("diff", lambda a, b: diff_if_not_string(a, b))
OmegaConf.register_new_resolver("flat", lambda s: s.replace("/", "+"))
OmegaConf.register_new_resolver("uuid", lambda kind="short": _uuid(kind))
OmegaConf.register_new_resolver("mod", lambda x, y: int(x) % int(y))
OmegaConf.register_new_resolver("int_div", lambda a, b: int(a) // int(b))
OmegaConf.register_new_resolver("prod", lambda *args: np.prod(args))

log = logging.getLogger(__name__)


class EntryInterface:
    """
        A high-level interface for managing machine learning experiments,
        data source discovery, configuration composition, and model training.
    """

    def __init__(self):
        """
            Initializes the EntryInterface, sets up internal configuration tracking,
            and populates the list of available data sources from the config path.
        """
        self._config_path = project_config.config_path

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

        log.info('Processing datasource.')

        preprocessor = InterfacePreProcessor(datasource,
                                             transforms,
                                             preprocessor_mode=None,
                                             cache_path=project_config.cache_path)

        return preprocessor.pipeline()

    @staticmethod
    @lru_cache
    def get_available_datasources():
        return EntryInterface.get_available_resource('datasource')

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

            resources = EntryInterface.get_available_resource(str(resource_type),
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
    def get_available_resource(resource_type: str, max_level: int = None, as_tuple: bool = False):
        """
            Scans the internal configuration directory to list all available resource files
            for a specific type.

            Args:
                resource_type (str): The subdirectory to scan (e.g., 'model', 'datasource').
                max_level (int, optional): The maximum directory depth to search.

            Returns:
                list[str]: A list of relative paths (identifiers) for available resources.
        """

        if max_level is None or max_level <= 0:
            max_level = np.inf

        path = Path(project_config.config_path)
        path = path / resource_type

        all_files = []

        level = 0

        for root, dirs, files in os.walk(path):
            for file in files:
                all_files.append(str((Path(root) / file).relative_to(path)))

            level += 1
            if level > max_level:
                break

        if as_tuple:
            return list(map(lambda x: x[0] if len(x) == 1 else x,
                            [tuple(td.replace('.yaml', '').split(os.sep)) for td in all_files]))

        if resource_type == 'datasource':
            all_files = [f.replace('.yaml', '') for f in all_files]

        return all_files

    def get_model_cfg(self, model: AbsModelConfig | CustomModelTrainer | str | dict, task: str) :

        if isinstance(model, dict):
            return model, None
        elif isinstance(model, CustomModelTrainer):
            log.info('Using user defined model.')
            return model, None
        elif isinstance(model, (AbsModelConfig, str)):
            model_folder_name = model.config_name if not isinstance(model, str) else model

            log.info(f'Using a picid model config trainer for {model}.')

            available_model_definition_as_tuples = self.get_available_resource(f'model_configs/{task}', as_tuple=True)
            assert model_folder_name in available_model_definition_as_tuples, (
                f'Selected model definition {model_folder_name} '
                'does not exist. Select one from the '
                'following list:'
                f' {available_model_definition_as_tuples}')

            model_config_path = (project_config.config_path / f'model_configs/{task}' / (model_folder_name + '.yaml'))
            model_cfg = OmegaConf.load(model_config_path)

            if isinstance(model, AbsModelConfig):
                model_cfg.update(model.model_dump(by_alias=True))

            log.info('Using a picid model anmd its associated trainer.')
            log.info(f'Model path: {model_config_path}')

            return model_cfg, model_folder_name
        else:
            assert False, (f'Invalid model type {type(model)}. Allowed types are:'
                           ' AbsModelConfig | CustomModelTrainer | str | dict.')

    def get_task_definition_cfg(self, task_definition: BaseTaskDefinition | tuple[str, str]):

        if isinstance(task_definition, dict):
            task_definition_cfg = task_definition
            task_folder_name = None

        elif isinstance(task_definition, (tuple, str)):

            task_folder_name = task_definition[0]

            available_task_definition_as_tuples = self.get_available_resource('task_definition',
                                                                              as_tuple=True)

            assert task_definition in available_task_definition_as_tuples, (
                f'Selected task definition {task_folder_name} does not exist. '
                f'Select one from the following list: {available_task_definition_as_tuples}')

            selected_task_definition = os.sep.join(task_definition) + '.yaml'
            task_definition_path = project_config.config_path / 'task_definition' / selected_task_definition

            task_definition_cfg = OmegaConf.load(task_definition_path)
        elif isinstance(task_definition, BaseTaskDefinition):
            task_definition_cfg = task_definition.model_dump(by_alias=True)
            task_folder_name = task_definition.config_name
        else:
            assert False, (f'Invalid task definition {type(task_definition)}. '
                           f'Allowed types are: BaseTaskDefinition | tuple[str, str]')

        return task_definition_cfg, task_folder_name

    def get_datasource_cfg(self, datasource: str | tuple[str, str] | dict,
                           task: str = None,
                           model: str = None):

        if isinstance(datasource, tuple):
            source, sub_folder = datasource

            log.info('Datasource as a tuple is used. The complete experiment config will be loaded from disk.')

            assert model is not None, \
                "When using datasource as a tuple you must specify also the model to use."

            available_experiments = self.get_available_resource(f'experiment/{source}', as_tuple=True)
            available_experiments = [t for t in available_experiments if isinstance(t, tuple) and t[-1] != 'base']

            available_tasks = [t[0] for t in available_experiments]

            assert task in available_tasks, (f"Datasource {source} does not have a default experiment for task {task}. "
                                             f"Available tasks are: {available_tasks}.")

            available_sub_folder = [t[1] for t in available_experiments if len(t) > 1 and t[:2] == (task, sub_folder)]

            assert sub_folder in available_sub_folder, (f"Datasource {source} and task {task} do not have a default experiment value {sub_folder}. "
                                             f"Available selections are: {available_sub_folder}.")

            available_models = [t[2] for t in available_experiments if len(t) > 1 and t[:2] == (task, sub_folder)]

            assert model in available_models, (f"Datasource {datasource} and task {task} do not have a model {model}. "
                                             f"Available models are: {available_models}.")

            complete_path = (source, task, sub_folder, model)

            _complete_path = project_config.config_path / 'experiment' / ('/'.join(complete_path) + '.yaml')

            datasource_cfg = OmegaConf.load(_complete_path)
            OmegaConf.update(datasource_cfg, 'is_experiment', True)

        elif isinstance(datasource, str):
            datasource_cfg = self.load_resource(datasource, 'datasource')
        elif isinstance(datasource, dict):
            datasource_cfg = datasource
        else:
            assert False, (f'Invalid datasource type {type(datasource)}. '
                           f'Allowed types are: datasource: str | tuple[str, str] | dict')

        return datasource_cfg

    def _compose_config_file(self,
                             *,
                             run_name: str,
                             task_definition: BaseTaskDefinition | tuple[str, str] | str | None,
                             model: AbsModelConfig | CustomModelTrainer | str | None,
                             datasource: str | tuple[str, str] | ProcessedDatasource | None,
                             evaluators: list[str] | str | dict = None,
                             callbacks: list[str] | str | dict | Callback | list[Callback] | None = None,
                             transforms: str | dict = None,
                             overrides: List[str] | str = None
                             ) -> DictConfig:
        
        """
            Assembles a full Hydra experiment configuration by merging model,
            task, data, and callback configurations.

            Args:
                task_definition (BaseTaskDefinition | str): Definition of the learning task.
                model (AbsModelConfig): Configuration for the architecture.
                datasource (str | ProcessedDatasource): Data identifier or processed object.
                callbacks: Configuration for training callbacks.
                transforms: Data transformation configurations.
                overrides (List[str]): Manual Hydra overrides (e.g., ['+params.lr=0.01']).

            Returns:
                DictConfig: A unified configuration object ready for Hydra instantiation.
        """

        # TODO: CHECK transforms
        # TODO: CHECK evaluators

        log.info('Composing internal experiment components.')

        model_cfg = None
        datasource_cfg = None
        task_cfg = None

        if overrides is None: overrides = []

        if isinstance(overrides, str):
            overrides = [overrides]

        # task_definition must be a tuple because we need to load it from task_definitions folder in configs
        # alternatively it can be a BaseTaskDefinition
        if task_definition is not None:
            task_cfg, task_yaml_name = self.get_task_definition_cfg(task_definition)

        # If the model is not a custom one, it must be loaded from disk with the associated from model_configs folder
        if model is not None:
            if isinstance(model, CustomModelTrainer):
                model_cfg, model_yaml_name = None, None
            else:
                model_cfg, model_yaml_name = self.get_model_cfg(model, task=task_yaml_name)

        if datasource is not None:
            if isinstance(datasource, ProcessedDatasource):
                datasource_cfg = None
                overrides.append(f'+datasource.task_mode={datasource.task_mode}')
                log.info(f'Using custom datasource as override (+datasource.task_mode={datasource.task_mode})')
            else:
                datasource_cfg = self.get_datasource_cfg(datasource,
                                                         task=task_yaml_name,
                                                         model=model_yaml_name)

        # local_instance = GlobalHydra.instance()
        # GlobalHydra.instance().clear()

        cs = ConfigStore.instance()

        if datasource_cfg is not None:
            if datasource_cfg.get('is_experiment', False):
                cs.store(group="experiment",
                         name="experiment_cfg",
                         package="_global_",
                         node=datasource_cfg)
                overrides.append('experiment=experiment_cfg')
                log.info(f'Using experiment {datasource}')
            else:
                cs.store(group="datasource",
                         name="datasource_cfg",
                         node=datasource_cfg)
                overrides.append('datasource=datasource_cfg')
                log.info(f'Using picid datasource {datasource}')

        if model_cfg is not None:
            cs.store(group="model_configs", name="model_configs_cfg", package="_global_",
                     node=model_cfg)
            overrides.append('+model_configs=model_configs_cfg')
        elif isinstance(model, CustomModelTrainer):
            assert task_yaml_name is not None, f'When using a custom model, task_definition must be provide and cannot be a dictionary {type(task_definition)} '
            custom_path = project_config.config_path / 'model_configs' / task_yaml_name / 'custom_base.yaml'
            training_model_cfg = OmegaConf.load(custom_path)

            cs.store(group="model_configs", name="model_configs_cfg", package="_global_",
                     node=training_model_cfg)
            overrides.append('+model_configs=model_configs_cfg')
        
        if task_cfg is not None:
            cs.store(group="task_definition",
                     name="task_definition_cfg",
                     node=task_cfg)

            overrides.append('task_definition=task_definition_cfg')

        if callbacks is not None:
            if callbacks == 'default':
                log.info('Using custom callbacks.')
                callbacks_cfg = OmegaConf.load(project_config.config_path / 'callbacks' / 'default.yaml')

                cs.store(group="callbacks",
                         name="callbacks_cfg",
                         node=callbacks_cfg)
                overrides.append('callbacks=callbacks_cfg')
            else:
                if not isinstance(callbacks, Sequence):
                    callbacks = [callbacks]

                callbacks_to_inst = len(set(type(c) for c in callbacks
                                            if not isinstance(c, (DeprecatedPytorchCallback, PytorchCallback))))

                if callbacks_to_inst > 0:

                    log.info('Creating callbacks confing.')

                    assert all(isinstance(c, (str, dict)) for c in callbacks), \
                        ('In addition to lightning callbacks, you can defined them as string or dicts. '
                         f'{[isinstance(c, (str, dict)) for c in callbacks]} where given.')

                    callbacks_cfg = []
                    for c in callbacks:
                        if isinstance(c, str):
                            callbacks_cfg.append(OmegaConf.load(project_config.config_path / 'callbacks' / f'{c}.yaml'))
                        elif isinstance(callbacks, dict):
                            callbacks_cfg.append(c)

                    cs.store(group="callbacks",
                             name="callbacks_cfg",
                             node=callbacks_cfg)
                    overrides.append('callbacks=callbacks_cfg')
                else:
                    log.info('No callbacks for creating a confing.')

        # TODO: FIX AND TEST
        if evaluators is not None or isinstance(evaluators, dict):
            if isinstance(evaluators, dict):
                evaluators_cfg = {s: v.model_dump(by_alias=True) for s, v in evaluators.items()}
                log.info('Using custom evaluators.')
            else:
                # FIX IT
                evaluators_cfg = OmegaConf.load(project_config.config_path / 'evaluator' / 'rul.yaml')
                log.info('Using default evaluators.')

            cs.store(group="evaluator",
                     name="evaluator_cfg",
                     node=evaluators_cfg)
            overrides.append('evaluator=evaluator_cfg')

        cs.store(group="paths",
                 name="paths_cfg",
                 node=project_config.model_dump())
        overrides.append('paths=paths_cfg')

        hydra_config = OmegaConf.load(project_config.config_path / 'hydra' / 'interface.yaml')
        # OmegaConf.update(hydra_config, 'interface_run_name', run_name)

        cs.store(group="hydra",
                 name="hydra_cfg",
                 node=hydra_config)
        overrides.append('hydra=hydra_cfg')

        overrides.append(f'hydra.job.name={run_name}')


        cfg = hydra.compose(config_name='run',
                            overrides=overrides,
                            return_hydra_config=True)

        # display_targets(cfg)

        # GlobalHydra.set_instance(local_instance)

        return cfg


    def train(self, *,
              run_name: str,
              model: AbsModelConfig | CustomModelTrainer | str,
              task_definition: BaseTaskDefinition | tuple[str, str] | str,
              datasource: ProcessedDatasource | tuple[str, str] | str | None,
              callbacks: Callback | list[Callback] | dict | None = None,
              loggers: list[Logger | BaseLogger] = None,
              evaluators: dict[str, DefaultEvaluatorConfig] = None,
              transforms: list[DataTransform] = None,
              overrides: List[str] | str = None,
              enable_progress_bar: bool = True,
              debug: bool = False,
              seed: int = None) -> None:

        """
            The main entry point for running an experiment. It handles data loading,
            preprocessing, dataset instantiation, and PyTorch Lightning trainer execution.

            Args:
                task_definition (BaseTaskDefinition | tuple[str, str]): Definition of the task and its metrics. it can be defined by the user using a BaseTaskDefinition config file or passed a tuple of string (task_name, task_type)
                model (AbsModelConfig): The model configuration to use.
                datasource: The source data identifier.
                callbacks: Training callbacks.
                loggers (list): List of loggers for tracking metrics (e.g., TensorBoard, WandB).
                transforms (list): Data transformations to apply before training. It is ignored if datasource isinstance of ProcessedDatasource.
                overrides (list): Hydra configuration overrides. It must respect the Hydra overriding syntax.
                enable_progress_bar (bool): Whether to show the Lightning progress bar.

            Returns:
                None: This function initiates the `trainer.fit()` process.
        """

        local_instance = GlobalHydra.instance()
        GlobalHydra.instance().clear()

        with hydra.initialize_config_dir(config_dir=str(project_config.config_path.absolute()),
                                         job_name="interface", version_base="1.3"):

            if transforms is not None:
                assert isinstance(transforms, Sequence), ('transforms must be a list of transforms '
                                                          'or None (default one will be used, if possible).')

            if debug:
                log.info('Debug mode is enabled. No training will be performed.')

            cfg = self._compose_config_file(run_name = run_name,
                                            datasource=datasource,
                                            task_definition=task_definition,
                                            evaluators=evaluators,
                                            callbacks=callbacks,
                                            overrides=overrides,
                                            model=model)

            HydraConfig.instance().set_config(cfg)
            log.info(yaml.dump(OmegaConf.to_container(cfg, resolve=False), default_flow_style=False, indent=4))
            # display_targets(cfg)

            # if cfg.get("num_threads", None) is not None:
            #     log.warning(f"Setting number of threads for all libraries to {cfg.num_threads}")
            #     verify_thread_limits(cfg.num_threads)
            #     parallel_backend("loky", n_jobs=cfg.num_threads)
            #     torch.set_num_threads(cfg.num_threads)
            #     torch.set_num_interop_threads(cfg.num_threads)
            #     log.info(f"Set number of threads to {cfg.num_threads}")
            # else:
            #     log.info("Using default number of threads for all libraries")

            if seed is None:
                seed = cfg.get('seed', seed)

            seed_everything(seed, workers=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

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

            log.info('Training config file creation ended.')

            if isinstance(datasource, ProcessedDatasource):
                data_dict = datasource.data_dict
                meta_data_dict = datasource.meta_data_dict

                log.info('Using an already processed datasource.')

            else:
                log.info('Loading the datasource.')
                # if isinstance(datasource, tuple):
                datasource = cfg.datasource
                # else:
                #     datasource = self.load_resource(datasource, 'datasource')

                dataset_source = hydra.utils.instantiate(datasource)

                log.info('Processing the datasource.')
                if transforms is None:
                    assert 'transforms' in cfg, ('There is not transforms defined in config file. '
                                                 'Fix the inputs of this function or pass a custom transforms list.')
                    transforms_cfg = hydra.utils.instantiate(cfg['transforms'])

                    transforms = []
                    for k, t in transforms_cfg.items():
                        transforms.append(DataTransform(transform_name=k, transform=t['transform'],
                                                        metadata=t['metadata']))

                process_datasource = self.process_datasource(dataset_source, transforms)

                data_dict = process_datasource.data_dict
                meta_data_dict = process_datasource.meta_data_dict

                log.info('Datasource correctly processed.')

            register_data_dim_resolver(data=data_dict["train"])

            features_set = set(data_dict["train"].keys())
            for split in ["val", "test"]:
                assert set(data_dict[split].keys()) == features_set, (
                    f"Mismatch in features for split {split}. "
                    f"Expected: {features_set}, "
                    f"Found: {set(data_dict[split].keys())}")

            input_data_task_keys = cfg.task_definition.model.data_requirements.input_tensors

            log.info('Creating the datasets.')

            datasets = {}
            for split in ["train", "val", "test"]:
                log.info(f'Creating {split} dataset.')

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

            log.info('Creating loaders.')

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

                log.info(f"[1/{loader_lengths[split]}] Batch shapes for {split}:")

                batch = next(iter(loader))

                for key, value in batch.items():
                    # TODO: handle nested structures (attributedict, list, etc)
                    if isinstance(value, torch.Tensor):
                        log.info(f"  {key}: {value.shape}")
                    else:
                        log.info(f"  {key}: Non-tensor value of type {type(value)}")

            log_functions = []

            if loggers is None:
                loggers = []

            log.info('Creating loggers.')

            for logger in loggers:
                if isinstance(logger, Logger):
                    log_functions.append(logger)
                elif isinstance(logger, BaseLogger):
                    d = logger.model_dump(by_alias=True)
                    d['save_dir'] = project_config.save_dir
                    log_functions.append(hydra.utils.instantiate(d))

            for d in cfg.get("logger", {}).values():
                log_functions.append(hydra.utils.instantiate(d))

            log.info('Creating evaluators.')

            register_infer_dataloader_length_resolver(loader_lengths)

            evaluators = {}
            if 'evaluator' in cfg and 'trasforms' in cfg:

                # TODO: try to remove it in favor of the custom define one
                transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)

                for s in ["train", "val", "test"]:
                    inverse_transform_required = cfg.evaluator[s].get("apply_inverse_scaling", False)
                    inverse_transform_key = cfg.evaluator[s].get("inverse_transform_key", None)
                    inverse_transform_which = cfg.evaluator[s].get("inverse_transform_which", "last")
                    inverse_transform_name = cfg.evaluator[s].get("inverse_transform_name", None)

                    inverse_transform = None
                    if inverse_transform_required:
                        # Prefer key-based lookup: discover inverter from registry
                        if inverse_transform_key is not None:
                            inverse_transform, chosen_name = (transforms_manager.get_inverter_for_key_with_name(
                                    inverse_transform_key, which=inverse_transform_which))

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
                    for lgr in log_functions:
                        if isinstance(lgr, WandbLogger):
                            remote_logger = lgr

                    evaluator = hydra.utils.instantiate(
                        cfg.evaluator[s],
                        inverse_transform=inverse_transform,
                        remote_logger=remote_logger,
                    )
                    log.info(f"Evaluator for {s}: {cfg.evaluator[s]._target_}")
                    evaluators[s] = evaluator

            else:
                for s in ["train", "val", "test"]:
                    evaluator = hydra.utils.instantiate(
                        cfg.evaluator[s],
                        inverse_transform=None,
                        remote_logger=None,
                    )
                    evaluators[s] = evaluator

            loss = hydra.utils.instantiate(cfg.loss)

            if callbacks is not None:
                if isinstance(callbacks, Callback) or (isinstance(callbacks, list) and isinstance(callbacks[0], Callback)):
                    log.info(f'Using user define callbacks: {callbacks}.')
            else:
                if 'callbacks' in cfg:
                    log.info('Loading callbacks.')
                    for _, k in cfg['callbacks'].items():
                        log.info(f"Instantiating callback <{k['_target_']}>")

                    callbacks: list[Callback] = list(hydra.utils.instantiate(cfg["callbacks"]).values())
                else:
                    log.info('No callbacks to load.')

            for callback in callbacks:
                if isinstance(callback, ModelCheckpointWithConfig):
                    callback.config = cfg

            if 'model' in cfg and 'config_name' in cfg.model:
                with open_dict(cfg):
                    del cfg.model['config_name']

            log.info('Creating torch lightning module.')
            model = create_lightning_module(
                backbone=model if isinstance(model, CustomModelTrainer) else None,
                cfg=cfg,
                datamodule=datamodule,
                evaluators=evaluators,
                loss=loss,
            )

            log.info('Creating the trainer class.')
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer,
                callbacks=callbacks,
                logger=log_functions,
                num_sanity_val_steps=0,
                # **additional_trainer_args,
                enable_progress_bar=enable_progress_bar,
            )

            if not debug:
                log.info('Fitting the trainer.')
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

                print(test_results)
                return test_results

        GlobalHydra.set_instance(local_instance)