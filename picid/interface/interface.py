"""High-level experiment interface for PICID: config composition, data preprocessing, and training."""

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
from rich.console import Console
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
from picid.interface.schemas.loggers import BaseLogger
from picid.interface.schemas.model import AbsModelConfig
from picid.interface.schemas.task_definition import BaseTaskDefinition
from picid.interface.schemas.trainer import TrainerConfig
from picid.interface.schemas.evaluators import AbsEvalConfig
from picid.interface.utils import (
    ProcessedDatasource,
    InterfacePreProcessor,
    register_data_dim_resolver,
    create_lightning_module,
    get_inverter_for_key_with_name,
)
from picid.transforms.base.data_transform import DataTransform
from picid.utils import print_hydra_config_tree
from picid.callbacks.model_checkpoint import ModelCheckpointWithConfig
from picid.transforms.base.multisource import InverseTransformMixin

log = logging.getLogger(__name__)


def verify_thread_limits(expected: int):
    """
    Log the expected thread count and all relevant thread-limit env vars.

    Parameters
    ----------
    expected : int
        The expected number of threads to log for sanity-checking.
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


def register_infer_dataloader_length_resolver(lengths: dict[str, int]) -> None:
    """
    Register an OmegaConf resolver that looks up dataloader lengths by split name.

    Parameters
    ----------
    lengths : dict[str, int]
        Mapping from split name (e.g. ``"train"``) to dataloader length.
    """

    def get_key(lengths, key):
        """
        Return the dataloader length for ``key``, raising KeyError when absent.

        Parameters
        ----------
        lengths : dict[str, int]
            Mapping of split names to dataloader lengths.
        key : str
            Split name to look up.

        Returns
        -------
        int
            Dataloader length for the requested split.
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


def safe_div(a, b):
    """
    Divide ``a`` by ``b``, raising if not exactly divisible.

    When ``b`` is an iterable (not str/bytes), tries each candidate in order
    and returns the first exact quotient; raises if none succeed.

    Parameters
    ----------
    a : int
        Dividend.
    b : int or Iterable[int]
        Divisor or ordered sequence of candidate divisors.

    Returns
    -------
    int
        Exact integer quotient ``a // b``.
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


def diff_if_not_string(a, b):
    """
    Return ``a - b``, or ``a`` unchanged when either operand is a string.

    Parameters
    ----------
    a : numeric or str
        Minuend.
    b : numeric or str
        Subtrahend.

    Returns
    -------
    numeric or str
        ``a - b`` when both are numeric, otherwise ``a``.
    """
    if isinstance(a, str) or isinstance(b, str):
        return a
    return a - b


def _uuid(kind: str) -> str:
    """
    Generate a UUID string.

    Parameters
    ----------
    kind : str
        Format selector: ``"hex"`` returns a 32-character hex string;
        ``"short"`` returns the first 8 characters.

    Returns
    -------
    str
        UUID string in the requested format.
    """
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


class PicidExperiment:
    """
    A high-level interface for managing machine learning experiments.

    This class provides a unified API for data source discovery, hierarchical
    configuration composition using Hydra/OmegaConf, and managing the
    end-to-end training lifecycle via PyTorch Lightning.
    """

    def get_datasource(self, v: str):
        """
        Load and instantiate a built-in data source by name.

        Parameters
        ----------
        v : str
            Name or path identifier of the data source (e.g. ``"phme20"``).
            Run ``get_available_datasources()`` to list valid names.

        Returns
        -------
        AbstractDataSourceLoader
            An instantiated data source loader, ready to pass to
            ``process_datasource()``.
        """
        datasource = self.load_resource(v, "datasource")
        dataset_source = hydra.utils.instantiate(datasource)

        return dataset_source

    def process_datasource(
        self,
        datasource: str | AbstractDataSourceLoader,
        transforms: list[DataTransform],
        cache: bool = True,
    ):
        """
        Apply a sequence of transforms to a datasource and return processed data.

        Parameters
        ----------
        datasource : str or AbstractDataSourceLoader
            Either a string name (resolved via ``get_datasource()``) or an
            already-instantiated loader such as ``CustomSingleSourceLoader``.
        transforms : list[DataTransform]
            Ordered list of transforms to apply. Transforms are fitted on the
            training split and applied to all splits.
        cache : bool, default True
            Cache the transformed dataset on disk under
            ``project_config.cache_path``. When ``False``, transforms run
            every call (joblib ``Memory(None)`` no-op).

        Returns
        -------
        ProcessedDatasource
            Wrapper around the transformed data, ready to pass as
            ``datasource=`` to ``train()``.
        """

        log.info("Processing datasource.")

        preprocessor = InterfacePreProcessor(
            datasource,
            transforms,
            preprocessor_mode=None,
            cache_path=project_config.cache_path if cache else None,
        )

        return preprocessor.pipeline()

    @staticmethod
    @lru_cache
    def get_available_datasources():
        """
        List all data sources available in the configuration directory.

        Returns
        -------
        list[str]
            Names of all registered datasources (each is a valid argument to
            ``get_datasource()``).
        """
        return PicidExperiment.get_available_resource("datasource")

    @staticmethod
    def load_resource(v: str | Path | dict = None, resource_type: str = None):
        """
        Resolve and load a configuration resource from disk or a dict.

        Parameters
        ----------
        v : str or Path or dict
            Resource definition. A ``dict`` is wrapped in ``OmegaConf.create``.
            A string or ``Path`` is looked up under
            ``configs/<resource_type>/``.
        resource_type : str
            Resource category, e.g. ``"datasource"``, ``"model"``,
            ``"evaluator"``.

        Returns
        -------
        DictConfig
            The loaded OmegaConf configuration object.

        Raises
        ------
        AssertionError
            If the string/Path does not match any registered resource.
        TypeError
            If ``v`` is not a ``str``, ``Path``, or ``dict``.
        """
        if isinstance(v, dict):
            cfg = OmegaConf.create(v)
        elif isinstance(v, (str, Path)):
            v = Path(v)
            root = project_config.config_path / resource_type

            if v.is_relative_to(root):
                v = v.relative_to(root)

            max_level = None
            if resource_type == "datasource":
                max_level = 1

            resources = PicidExperiment.get_available_resource(
                str(resource_type), max_level=max_level
            )

            v = str(v)

            assert v in resources, (
                f"Given path {v.replace('.yaml', '')} is not present in the resource type {resource_type}. "
                f"Available sources are {resources}."
            )

            if resource_type == "datasource":
                v = v + ".yaml"

            path = root / v
            cfg = OmegaConf.load(path)
        else:
            raise TypeError(
                f"load_resource expects a str, Path, or dict, got {type(v)}."
            )

        if resource_type == "datasource":
            cfg["cache_dir"] = project_config.cache_dir
            cfg["data_path"] = project_config.data_dir / cfg["_target_"].split(".")[-1]

        return cfg

    @staticmethod
    def get_available_resource(
        resource_type: str, max_level: int = None, as_tuple: bool = False
    ):
        """
        Scan the config directory and return available resource identifiers.

        Parameters
        ----------
        resource_type : str
            Subdirectory to scan under ``configs/`` (e.g. ``"datasource"``,
            ``"model"``, ``"evaluator"``).
        max_level : int or None
            Maximum directory depth to descend. ``None`` means unlimited.
        as_tuple : bool
            When ``True``, return each path as a tuple of path components
            rather than a joined string. Default ``False``.

        Returns
        -------
        list[str] or list[tuple[str, ...]]
            Relative paths (or tuples) of all YAML files found under the
            resource directory.
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
            return list(
                map(
                    lambda x: x[0] if len(x) == 1 else x,
                    [tuple(td.replace(".yaml", "").split(os.sep)) for td in all_files],
                )
            )

        if resource_type == "datasource":
            all_files = [f.replace(".yaml", "") for f in all_files]

        return all_files

    def get_model_cfg(
        self, model: AbsModelConfig | CustomModelTrainer | str | dict, task: str
    ):
        """
        Resolve a model definition to a DictConfig (or pass through a custom trainer).

        Parameters
        ----------
        model : AbsModelConfig or CustomModelTrainer or str or dict
            Model definition. Accepted forms:

            * ``AbsModelConfig`` — serialised and merged with the YAML baseline.
            * ``str`` — name of a YAML file in ``configs/model_configs/<task>/``.
            * ``CustomModelTrainer`` — returned as-is; no config is loaded.
            * ``dict`` — passed through as-is.
        task : str
            Task config name used to locate the model YAML
            (e.g. ``"prognostics"``).

        Returns
        -------
        tuple[DictConfig or CustomModelTrainer or dict, str or None]
            ``(config_or_instance, config_name)``. ``config_name`` is ``None``
            for ``CustomModelTrainer`` and raw-dict inputs.

        Raises
        ------
        AssertionError
            If a string model name is not found under ``configs/model_configs/<task>/``.
        TypeError
            If ``model`` is not one of the accepted types.
        """

        if isinstance(model, dict):
            return model, None
        elif isinstance(model, CustomModelTrainer):
            log.info("Using user defined model.")
            return model, None
        elif isinstance(model, (AbsModelConfig, str)):
            model_folder_name = (
                model.config_name if not isinstance(model, str) else model
            )

            log.info(f"Using a picid model config trainer for {model}.")

            available_model_definition_as_tuples = self.get_available_resource(
                f"model_configs/{task}", as_tuple=True
            )
            assert model_folder_name in available_model_definition_as_tuples, (
                f"Selected model definition {model_folder_name} "
                "does not exist. Select one from the "
                "following list:"
                f" {available_model_definition_as_tuples}"
            )

            model_config_path = (
                project_config.config_path
                / f"model_configs/{task}"
                / (model_folder_name + ".yaml")
            )
            model_cfg = OmegaConf.load(model_config_path)

            if isinstance(model, AbsModelConfig):
                model_cfg.update(model.model_dump(by_alias=True))

            log.info("Using a picid model and its associated trainer.")
            log.info(f"Model path: {model_config_path}")

            return model_cfg, model_folder_name
        else:
            raise TypeError(
                f"Invalid model type {type(model)}. "
                f"Allowed types are: AbsModelConfig | CustomModelTrainer | str | dict."
            )

    def get_task_definition_cfg(
        self, task_definition: BaseTaskDefinition | tuple[str, str]
    ):
        """
        Resolve a task definition to an OmegaConf DictConfig.

        Parameters
        ----------
        task_definition : BaseTaskDefinition or tuple[str, str] or dict
            The task to resolve. Accepted forms:

            * ``BaseTaskDefinition`` instance — serialised via ``model_dump``.
            * ``tuple[str, str]`` — hierarchical path into the
              ``configs/task_definition/`` directory
              (e.g. ``("prognostics", "rul")``).
            * ``dict`` — passed through as-is.

        Returns
        -------
        tuple[dict or DictConfig, str or None]
            A ``(config, config_name)`` pair. ``config_name`` is ``None``
            when a raw dict is supplied.

        Raises
        ------
        AssertionError
            If a tuple reference does not match any available task definition.
        TypeError
            If ``task_definition`` is not one of the accepted types.
        """

        if isinstance(task_definition, dict):
            task_definition_cfg = task_definition
            task_folder_name = None

        elif isinstance(task_definition, (tuple, str)):
            task_folder_name = task_definition[0]

            available_task_definition_as_tuples = self.get_available_resource(
                "task_definition", as_tuple=True
            )

            assert task_definition in available_task_definition_as_tuples, (
                f"Selected task definition {task_folder_name} does not exist. "
                f"Select one from the following list: {available_task_definition_as_tuples}"
            )

            selected_task_definition = os.sep.join(task_definition) + ".yaml"
            task_definition_path = (
                project_config.config_path
                / "task_definition"
                / selected_task_definition
            )

            task_definition_cfg = OmegaConf.load(task_definition_path)
        elif isinstance(task_definition, BaseTaskDefinition):
            task_definition_cfg = task_definition.model_dump(by_alias=True)
            task_folder_name = task_definition.config_name
        else:
            raise TypeError(
                f"Invalid task definition type {type(task_definition)}. "
                f"Allowed types are: BaseTaskDefinition | tuple[str, str]."
            )

        return task_definition_cfg, task_folder_name

    def get_datasource_cfg(
        self,
        datasource: str | tuple[str, str] | dict,
        task: str = None,
        model: str = None,
    ):
        """
        Resolve a datasource identifier to an OmegaConf DictConfig.

        Parameters
        ----------
        datasource : str or tuple[str, str] or dict
            Datasource definition. Accepted forms:

            * ``str`` — name of a YAML file in ``configs/datasource/``.
            * ``tuple[str, str]`` — ``(source, sub_folder)`` pair that resolves
              to a complete experiment config under
              ``configs/experiment/<source>/<task>/<sub_folder>/<model>.yaml``.
              Requires ``task`` and ``model`` to be provided.
            * ``dict`` — passed through as-is.
        task : str, optional
            Task config name. Required when ``datasource`` is a tuple.
        model : str, optional
            Model config name. Required when ``datasource`` is a tuple.

        Returns
        -------
        DictConfig
            The resolved datasource or full experiment configuration.

        Raises
        ------
        AssertionError
            If the experiment path does not exist or required arguments are missing.
        TypeError
            If ``datasource`` is not one of the accepted types.
        """

        if isinstance(datasource, tuple):
            source, sub_folder = datasource

            log.info(
                "Datasource as a tuple is used. The complete experiment config will be loaded from disk."
            )

            assert (
                model is not None
            ), "When using datasource as a tuple you must specify also the model to use."

            available_experiments = self.get_available_resource(
                f"experiment/{source}", as_tuple=True
            )
            available_experiments = [
                t
                for t in available_experiments
                if isinstance(t, tuple) and t[-1] != "base"
            ]

            available_tasks = [t[0] for t in available_experiments]

            assert task in available_tasks, (
                f"Datasource {source} does not have a default experiment for task {task}. "
                f"Available tasks are: {available_tasks}."
            )

            available_sub_folder = [
                t[1]
                for t in available_experiments
                if len(t) > 1 and t[:2] == (task, sub_folder)
            ]

            assert sub_folder in available_sub_folder, (
                f"Datasource {source} and task {task} do not have a default experiment value {sub_folder}. "
                f"Available selections are: {available_sub_folder}."
            )

            available_models = [
                t[2]
                for t in available_experiments
                if len(t) > 1 and t[:2] == (task, sub_folder)
            ]

            assert model in available_models, (
                f"Datasource {datasource} and task {task} do not have a model {model}. "
                f"Available models are: {available_models}."
            )

            complete_path = (source, task, sub_folder, model)

            _complete_path = (
                project_config.config_path
                / "experiment"
                / ("/".join(complete_path) + ".yaml")
            )

            datasource_cfg = OmegaConf.load(_complete_path)
            OmegaConf.update(datasource_cfg, "is_experiment", True)

        elif isinstance(datasource, str):
            datasource_cfg = self.load_resource(datasource, "datasource")
        elif isinstance(datasource, dict):
            datasource_cfg = datasource
        else:
            raise TypeError(
                f"Invalid datasource type {type(datasource)}. "
                f"Allowed types are: str | tuple[str, str] | dict."
            )

        return datasource_cfg

    def _compose_config_file(
        self,
        *,
        run_name: str,
        task_definition: BaseTaskDefinition | tuple[str, str] | str | None,
        model: AbsModelConfig | CustomModelTrainer | str | None,
        datasource: str | tuple[str, str] | ProcessedDatasource | None,
        evaluators: list[str] | str | dict = None,
        callbacks: list[str] | str | dict | Callback | list[Callback] | None = None,
        trainer_config: TrainerConfig | None = None,
        transforms: str | dict = None,
        overrides: List[str] | str = None,
    ) -> DictConfig:
        """
        Assemble a unified Hydra configuration from various components.

        Merges model, task, datasource, evaluator, and trainer configs
        into a single DictConfig object using the Hydra ConfigStore.

        Parameters
        ----------
        run_name : str
            Name of the current run.
        task_definition : BaseTaskDefinition or tuple[str, str] or str or None
            Task config or identifier.
        model : AbsModelConfig or CustomModelTrainer or str or None
            Model config or identifier.
        datasource : str or tuple[str, str] or ProcessedDatasource or None
            Datasource config or identifier.
        evaluators : list[str] or str or dict, optional
            Evaluation config.
        callbacks : list[str] or str or dict or Callback or list[Callback] or None
            Callback configs or instances.
        trainer_config : TrainerConfig or None
            Trainer parameters.
        transforms : str or dict, optional
            Data transformation configs.
        overrides : list[str] or str, optional
            Manual Hydra override strings.

        Returns
        -------
        DictConfig
            The fully composed Hydra configuration.
        """

        log.info("Composing internal experiment components.")

        model_cfg = None
        model_yaml_name = None
        datasource_cfg = None
        task_cfg = None

        if overrides is None:
            overrides = []

        if isinstance(overrides, str):
            overrides = [overrides]

        task_yaml_name = None
        # task_definition must be a tuple because we need to load it from task_definitions folder in configs
        # alternatively it can be a BaseTaskDefinition
        if task_definition is not None:
            task_cfg, task_yaml_name = self.get_task_definition_cfg(task_definition)

        # If the model is not a custom one, it must be loaded from disk with the associated from model_configs folder
        if model is not None:
            if isinstance(model, CustomModelTrainer):
                model_cfg, model_yaml_name = None, None
            else:
                if task_yaml_name is None:
                    raise ValueError("task_definition is none. You must define it.")
                model_cfg, model_yaml_name = self.get_model_cfg(
                    model, task=task_yaml_name
                )

        if datasource is not None:
            if isinstance(datasource, ProcessedDatasource):
                datasource_cfg = None
                overrides.append(f"+datasource.task_mode={datasource.task_mode}")
                log.info(
                    f"Using custom datasource as override (+datasource.task_mode={datasource.task_mode})"
                )
            else:
                datasource_cfg = self.get_datasource_cfg(
                    datasource, task=task_yaml_name, model=model_yaml_name
                )

        cs = ConfigStore.instance()

        if datasource_cfg is not None:
            if datasource_cfg.get("is_experiment", False):
                cs.store(
                    group="experiment",
                    name="experiment_cfg",
                    package="_global_",
                    node=datasource_cfg,
                )
                overrides.append("experiment=experiment_cfg")
                log.info(f"Using experiment {datasource}")
            else:
                cs.store(group="datasource", name="datasource_cfg", node=datasource_cfg)
                overrides.append("datasource=datasource_cfg")
                log.info(f"Using picid datasource {datasource}")

        if model_cfg is not None:
            cs.store(
                group="model_configs",
                name="model_configs_cfg",
                package="_global_",
                node=model_cfg,
            )
            overrides.append("+model_configs=model_configs_cfg")
        elif isinstance(model, CustomModelTrainer):
            assert (
                task_yaml_name is not None
            ), f"When using a custom model, task_definition must be provide and cannot be a dictionary {type(task_definition)} "
            custom_path = (
                project_config.config_path
                / "model_configs"
                / task_yaml_name
                / "custom_base.yaml"
            )
            training_model_cfg = OmegaConf.load(custom_path)

            cs.store(
                group="model_configs",
                name="model_configs_cfg",
                package="_global_",
                node=training_model_cfg,
            )
            overrides.append("+model_configs=model_configs_cfg")

        if task_cfg is not None:
            cs.store(group="task_definition", name="task_definition_cfg", node=task_cfg)

            overrides.append("task_definition=task_definition_cfg")

        if callbacks is not None:
            if callbacks == "default":
                log.info("Using custom callbacks.")
                callbacks_cfg = OmegaConf.load(
                    project_config.config_path / "callbacks" / "default.yaml"
                )

                cs.store(group="callbacks", name="callbacks_cfg", node=callbacks_cfg)
                overrides.append("callbacks=callbacks_cfg")
            else:
                if not isinstance(callbacks, Sequence):
                    callbacks = [callbacks]

                callbacks_to_inst = len(
                    set(
                        type(c)
                        for c in callbacks
                        if not isinstance(
                            c, (DeprecatedPytorchCallback, PytorchCallback)
                        )
                    )
                )

                if callbacks_to_inst > 0:
                    log.info("Creating callbacks confing.")

                    assert all(isinstance(c, (str, dict)) for c in callbacks), (
                        "In addition to lightning callbacks, you can defined them as string or dicts. "
                        f"{[isinstance(c, (str, dict)) for c in callbacks]} where given."
                    )

                    callbacks_cfg = []
                    for c in callbacks:
                        if isinstance(c, str):
                            callbacks_cfg.append(
                                OmegaConf.load(
                                    project_config.config_path
                                    / "callbacks"
                                    / f"{c}.yaml"
                                )
                            )
                        elif isinstance(callbacks, dict):
                            callbacks_cfg.append(c)

                    cs.store(
                        group="callbacks", name="callbacks_cfg", node=callbacks_cfg
                    )
                    overrides.append("callbacks=callbacks_cfg")
                else:
                    log.info("No callbacks for creating a confing.")

        if trainer_config is None:
            trainer_cfg = OmegaConf.load(
                project_config.config_path / "trainer" / "default.yaml"
            )
        else:
            trainer_cfg = trainer_config.model_dump(by_alias=True)

        cs.store(group="trainer", name="trainer_cfg", node=trainer_cfg)
        overrides.append("trainer=trainer_cfg")

        assert (
            isinstance(evaluators, str)
            or isinstance(evaluators, AbsEvalConfig)
            or evaluators is None
        ), f"Evaluators can be a string or a list of AbsEvalConfig or None. {type(evaluators)} where given"

        if evaluators is not None:
            if isinstance(evaluators, str):
                available_evaluators = self.get_available_resource(
                    "evaluator", as_tuple=True
                )

                assert (
                    evaluators in available_evaluators
                ), f"evaluators must be in [{available_evaluators}]. {evaluators} given."

                evaluators_cfg = OmegaConf.load(
                    project_config.config_path / "evaluator" / f"{evaluators}.yaml"
                )
            else:
                evaluators_cfg = evaluators.model_dump(by_alias=True)

            cs.store(group="evaluator", name="evaluator_cfg", node=evaluators_cfg)
            overrides.append("evaluator=evaluator_cfg")

        cs.store(group="paths", name="paths_cfg", node=project_config.model_dump())
        overrides.append("paths=paths_cfg")

        hydra_config = OmegaConf.load(
            project_config.config_path / "hydra" / "interface.yaml"
        )
        # OmegaConf.update(hydra_config, 'interface_run_name', run_name)

        cs.store(group="hydra", name="hydra_cfg", node=hydra_config)
        overrides.append("hydra=hydra_cfg")

        overrides.append(f"hydra.job.name={run_name}")

        cfg = hydra.compose(
            config_name="run", overrides=overrides, return_hydra_config=True
        )

        return cfg

    def train(
        self,
        *,
        run_name: str,
        model: AbsModelConfig | CustomModelTrainer | str,
        task_definition: BaseTaskDefinition | tuple[str, str] | str,
        training_config: TrainerConfig | None = None,
        datasource: ProcessedDatasource | tuple[str, str] | str | None,
        callbacks: Callback | list[Callback] | dict | None = None,
        loggers: list[Logger | BaseLogger] = None,
        evaluators: str = None,
        transforms: list[DataTransform] = None,
        overrides: List[str] | str = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        seed: int = None,
    ) -> None:
        """
        Run a complete training experiment.

        Orchestrates the full pipeline: Hydra config composition → data
        preprocessing → dataset/datamodule creation → logger and evaluator
        setup → Lightning module construction → ``trainer.fit()`` →
        ``trainer.test()``.

        Parameters
        ----------
        run_name : str
            Unique name for this experiment run (used as the job name in Hydra
            and the log directory).
        model : AbsModelConfig or CustomModelTrainer or str
            Model to train. See :meth:`get_model_cfg` for accepted forms.
        task_definition : BaseTaskDefinition or tuple[str, str]
            Task configuration. See :meth:`get_task_definition_cfg`.
        training_config : TrainerConfig, optional
            Trainer overrides. When ``None``, the project default is used.
        datasource : str or tuple[str, str] or ProcessedDatasource
            Data to train on. A ``ProcessedDatasource`` (from
            :meth:`process_datasource`) skips inline preprocessing.
        callbacks : Callback or list[Callback] or None
            PyTorch Lightning callbacks.
        loggers : list[Logger or BaseLogger], optional
            Loggers to attach to the trainer.
        evaluators : dict[str, AbsEvalConfig] or str or None
            Per-split evaluator configs keyed by ``"train"``, ``"val"``,
            ``"test"``. A string loads a built-in evaluator YAML by name.
        transforms : list[DataTransform], optional
            Preprocessing steps. Required when ``datasource`` is not already
            a ``ProcessedDatasource`` and the datasource config defines no
            transforms of its own.
        overrides : list[str] or str, optional
            Raw Hydra override strings (e.g. ``["trainer.max_epochs=50"]``).
        enable_progress_bar : bool
            Show the Lightning progress bar. Default ``True``.
        debug : bool
            Skip ``trainer.fit()`` and ``trainer.test()``. Useful for
            verifying config composition without running training.
            Default ``False``.
        seed : int, optional
            Random seed for reproducibility. Falls back to the config value
            when ``None``.

        Returns
        -------
        list[dict] or None
            Test results returned by ``trainer.test()``, or ``None`` when
            ``debug=True``.
        """

        local_instance = GlobalHydra.instance()
        GlobalHydra.instance().clear()

        with hydra.initialize_config_dir(
            config_dir=str(project_config.config_path.absolute()),
            job_name="interface",
            version_base="1.3",
        ):
            if transforms is not None:
                assert isinstance(transforms, Sequence), (
                    "transforms must be a list of transforms "
                    "or None (default one will be used, if possible)."
                )

            if debug:
                log.info("Debug mode is enabled. No training will be performed.")

            cfg = self._compose_config_file(
                run_name=run_name,
                trainer_config=training_config,
                datasource=datasource,
                task_definition=task_definition,
                evaluators=evaluators,
                callbacks=callbacks,
                overrides=overrides,
                model=model,
            )

            HydraConfig.instance().set_config(cfg)
            log.info(
                yaml.dump(
                    OmegaConf.to_container(cfg, resolve=False),
                    default_flow_style=False,
                    indent=4,
                )
            )
            tree = print_hydra_config_tree(cfg)
            Console().print(tree)
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
                seed = cfg.get("seed", seed)

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

            log.info("Training config file creation ended.")

            if isinstance(datasource, ProcessedDatasource):
                data_dict = datasource.data_dict
                meta_data_dict = datasource.meta_data_dict

                log.info("Using an already processed datasource.")

            else:
                log.info("Loading the datasource.")
                datasource = cfg.datasource

                dataset_source = hydra.utils.instantiate(datasource)

                log.info("Processing the datasource.")
                if transforms is None:
                    assert "transforms" in cfg, (
                        "There is not transforms defined in config file. "
                        "Fix the inputs of this function or pass a custom transforms list."
                    )
                    transforms_cfg = hydra.utils.instantiate(cfg["transforms"])

                    transforms = []
                    for k, t in transforms_cfg.items():
                        transforms.append(
                            DataTransform(
                                transform_name=k,
                                transform=t["transform"],
                                metadata=t["metadata"],
                            )
                        )
                else:
                    assert all(
                        isinstance(t, DataTransform) for t in transforms
                    ), f"transforms must be a list of DataTransform or None. {[type(t) for t in transforms]} were given."

                process_datasource = self.process_datasource(dataset_source, transforms)

                data_dict = process_datasource.data_dict
                meta_data_dict = process_datasource.meta_data_dict

                log.info("Datasource correctly processed.")

            register_data_dim_resolver(data=data_dict["train"])

            features_set = set(data_dict["train"].keys())
            for split in ["val", "test"]:
                assert set(data_dict[split].keys()) == features_set, (
                    f"Mismatch in features for split {split}. "
                    f"Expected: {features_set}, "
                    f"Found: {set(data_dict[split].keys())}"
                )

            input_data_task_keys = (
                cfg.task_definition.model.data_requirements.input_tensors
            )

            log.info("Creating the datasets.")

            datasets = {}
            for split in ["train", "val", "test"]:
                log.info(f"Creating {split} dataset.")

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
                    data_dict={
                        key: data_dict[split][key] for key in input_data_task_keys
                    },
                    meta_data_dict=meta_data_dict,
                )

            dataset_train = datasets["dataset_train"]
            dataset_val = datasets["dataset_val"]
            dataset_test = datasets["dataset_test"]

            datamodule = hydra.utils.instantiate(
                cfg.datamodule,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
            )

            log.info("Creating loaders.")

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

            log.info("Creating loggers.")

            for logger in loggers:
                if isinstance(logger, Logger):
                    log_functions.append(logger)
                elif isinstance(logger, BaseLogger):
                    d = logger.model_dump(by_alias=True)
                    d["save_dir"] = project_config.save_dir
                    log_functions.append(hydra.utils.instantiate(d))

            if not loggers:
                for d in cfg.get("logger", {}).values():
                    log_functions.append(hydra.utils.instantiate(d))

            log.info("Creating evaluators.")

            register_infer_dataloader_length_resolver(loader_lengths)

            evaluators = {}

            # if 'evaluator' in cfg:
            # TODO: try to remove it in favor of the custom define one
            # transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)

            for s in ["train", "val", "test"]:
                inverse_transform = None
                if cfg.evaluator[s].get("apply_inverse_scaling", False):
                    inverse_transform_key = cfg.evaluator[s].get(
                        "inverse_transform_key", None
                    )
                    inverse_transform_name = cfg.evaluator[s].get(
                        "inverse_transform_name", None
                    )
                    inverse_transform_which = cfg.evaluator[s].get(
                        "inverse_transform_which", "last"
                    )
                    # Prefer key-based lookup: discover inverter from registry

                    if inverse_transform_key is not None:
                        inverse_transform, chosen_name = get_inverter_for_key_with_name(
                            transforms,
                            inverse_transform_key,
                            which=inverse_transform_which,
                        )

                        # inverse_transform, chosen_name = (transforms_manager.get_inverter_for_key_with_name(
                        #         inverse_transform_key, which=inverse_transform_which))

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
                        inverse_transform = [
                            dt.transform_instance
                            for dt in transforms
                            if dt.name == inverse_transform_name
                        ]

                        inverse_transform = inverse_transform[0]

                        # inverse_transform = transforms_manager.get_transform(
                        #     inverse_transform_name
                        # ).transform_instance
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

            # else:
            #     for s in ["train", "val", "test"]:
            #         evaluator = hydra.utils.instantiate(
            #             cfg.evaluator[s],
            #             inverse_transform=None,
            #             remote_logger=None,
            #         )
            #         evaluators[s] = evaluator

            loss = hydra.utils.instantiate(cfg.loss)

            if callbacks is not None:
                if isinstance(callbacks, Callback):
                    log.info(f"Using user define callbacks: {callbacks}.")
                    callbacks = [callbacks]
                elif isinstance(callbacks, list) and isinstance(callbacks[0], Callback):
                    log.info(f"Using user define callbacks: {callbacks}.")
            else:
                if "callbacks" in cfg:
                    log.info("Loading callbacks.")
                    for _, k in cfg["callbacks"].items():
                        log.info(f"Instantiating callback <{k['_target_']}>")

                    callbacks: list[Callback] = list(
                        hydra.utils.instantiate(cfg["callbacks"]).values()
                    )
                else:
                    log.info("No callbacks to load.")
                    callbacks = []

            for callback in callbacks:
                if isinstance(callback, ModelCheckpointWithConfig):
                    callback.config = cfg

            if "model" in cfg and "config_name" in cfg.model:
                with open_dict(cfg):
                    del cfg.model["config_name"]

            log.info("Creating torch lightning module.")
            model = create_lightning_module(
                backbone=model if isinstance(model, CustomModelTrainer) else None,
                cfg=cfg,
                datamodule=datamodule,
                evaluators=evaluators,
                loss=loss,
            )

            log.info("Creating the trainer class.")
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer,
                callbacks=callbacks,
                logger=log_functions,
                num_sanity_val_steps=0,
                # **additional_trainer_args,
                enable_progress_bar=enable_progress_bar,
            )

            if not debug:
                log.info("Fitting the trainer.")
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
                    log.info(
                        "Starting testing without training (used for pre-trained models)!"
                    )
                    test_results = trainer.test(model=model, datamodule=datamodule)

                print(test_results)
                return test_results

        GlobalHydra.set_instance(local_instance)
