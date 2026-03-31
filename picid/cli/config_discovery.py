"""Config discovery: scan configs/ for task definitions, model configs, and experiments."""

from pathlib import Path
from typing import NamedTuple

CONFIGS_ROOT = Path(__file__).resolve().parent.parent.parent / "configs"


class TaskInfo(NamedTuple):
    """Task definition metadata."""

    task_type: str
    name: str
    path: Path


class DatasetGroupInfo(NamedTuple):
    """Dataset group metadata (top-level dir under experiment/)."""

    name: str
    path: Path


class ModelInfo(NamedTuple):
    """Model config metadata."""

    task_type: str
    model: str
    path: Path


def list_task_definitions() -> list[TaskInfo]:
    """
    Scan ``configs/task_definition`` and return entries for each YAML file.

    Returns
    -------
    list[TaskInfo]
        Task definitions discovered on disk.
    """
    root = CONFIGS_ROOT / "task_definition"
    if not root.exists():
        return []
    result: list[TaskInfo] = []
    for task_type_dir in sorted(root.iterdir()):
        if not task_type_dir.is_dir():
            continue
        task_type = task_type_dir.name
        for p in sorted(task_type_dir.glob("*.yaml")):
            result.append(TaskInfo(task_type=task_type, name=p.stem, path=p))
    return result


def list_model_configs(task_type: str | None = None) -> list[ModelInfo]:
    """
    Scan ``configs/model_configs`` and return entries for each YAML file.

    Parameters
    ----------
    task_type : str | None, default=None
        Optional task type filter.

    Returns
    -------
    list[ModelInfo]
        Model configs discovered on disk.
    """
    root = CONFIGS_ROOT / "model_configs"
    if not root.exists():
        return []
    result: list[ModelInfo] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        tt = task_dir.name
        if task_type is not None and tt != task_type:
            continue
        for p in sorted(task_dir.glob("*.yaml")):
            result.append(ModelInfo(task_type=tt, model=p.stem, path=p))
    return result


def list_experiment_groups() -> list[DatasetGroupInfo]:
    """
    Return top-level directories in ``configs/experiment``.

    Returns
    -------
    list[DatasetGroupInfo]
        Experiment groups discovered on disk.
    """
    root = CONFIGS_ROOT / "experiment"
    if not root.exists():
        return []
    result: list[DatasetGroupInfo] = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            result.append(DatasetGroupInfo(name=d.name, path=d))
    return result


def list_experiments_for_group_task(group: str, task_type: str) -> list[str]:
    """
    List experiment keys under ``experiment/<group>/<task_type>``.

    Parameters
    ----------
    group : str
        Dataset group name.
    task_type : str
        Task type directory name.

    Returns
    -------
    list[str]
        Experiment keys rooted at the given group and task type.
    """
    exp_root = CONFIGS_ROOT / "experiment" / group
    if not exp_root.exists():
        return []
    task_dir = exp_root / task_type
    if not task_dir.exists():
        return []
    result: list[str] = []
    for p in task_dir.rglob("*.yaml"):
        rel = p.relative_to(exp_root)
        key = str(rel.with_suffix("")).replace("\\", "/")
        result.append(f"{group}/{key}")
    return sorted(result)


def list_paths_configs() -> list[str]:
    """
    Return the stem of each YAML file in ``configs/paths``.

    Returns
    -------
    list[str]
        Available path config names.
    """
    root = CONFIGS_ROOT / "paths"
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.yaml"))


def list_debug_configs() -> list[str]:
    """
    Return the stem of each YAML file in ``configs/debug``.

    Returns
    -------
    list[str]
        Available debug config names.
    """
    root = CONFIGS_ROOT / "debug"
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.yaml"))


def get_task_type_from_experiment(experiment_key: str) -> str | None:
    """
    Extract the task type segment from an experiment key.

    Parameters
    ----------
    experiment_key : str
        Experiment key in ``group/task_type/...`` format.

    Returns
    -------
    str | None
        Task type segment, or ``None`` when the key is malformed.
    """
    parts = experiment_key.split("/")
    if len(parts) < 2:
        return None
    return parts[1]
