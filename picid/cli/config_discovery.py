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
    """Scan configs/task_definition/ and return TaskInfo for each YAML file."""
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
    """Scan configs/model_configs/ and return ModelInfo for each YAML file.

    If task_type is given, only return configs under that task type.
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
    """Return top-level dirs in configs/experiment/."""
    root = CONFIGS_ROOT / "experiment"
    if not root.exists():
        return []
    result: list[DatasetGroupInfo] = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            result.append(DatasetGroupInfo(name=d.name, path=d))
    return result


def list_experiments_for_group_task(group: str, task_type: str) -> list[str]:
    """List experiment keys under experiment/<group>/<task_type>/.

    experiment_key is the full path including group, e.g. unibo/prognostics/raw/cnn_1d.
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


def get_task_type_from_experiment(experiment_key: str) -> str | None:
    """Extract task type from experiment key.

    experiment_key format: group/task_type/... (e.g. unibo/prognostics/raw/cnn_1d).
    Returns the task_type segment (e.g. prognostics).
    """
    parts = experiment_key.split("/")
    if len(parts) < 2:
        return None
    return parts[1]
