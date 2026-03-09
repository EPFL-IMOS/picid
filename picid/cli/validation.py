"""Validation: task-model compatibility checks."""

from picid.cli.config_discovery import ModelInfo


def is_model_compatible_with_task(model: ModelInfo, task_type: str) -> bool:
    """Return True if the model is compatible with the given task type.

    A model is compatible when its task_type matches (i.e. it lives under
    model_configs/<task_type>/).
    """
    return model.task_type == task_type


def filter_models_for_task(models: list[ModelInfo], task_type: str) -> list[ModelInfo]:
    """Return only models compatible with the given task type."""
    return [m for m in models if is_model_compatible_with_task(m, task_type)]
