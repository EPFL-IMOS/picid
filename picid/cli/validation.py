"""Validation helpers for task-model compatibility."""

from picid.cli.config_discovery import ModelInfo


def is_model_compatible_with_task(model: ModelInfo, task_type: str) -> bool:
    """
    Check whether a model can be used for a task type.

    Parameters
    ----------
    model : ModelInfo
        Model metadata entry to inspect.
    task_type : str
        Task type requested by the experiment configuration.

    Returns
    -------
    bool
        ``True`` when the model belongs to the requested task type.
    """
    return model.task_type == task_type


def filter_models_for_task(models: list[ModelInfo], task_type: str) -> list[ModelInfo]:
    """
    Filter a list of models down to those compatible with a task type.

    Parameters
    ----------
    models : list[ModelInfo]
        Model metadata entries to filter.
    task_type : str
        Task type requested by the experiment configuration.

    Returns
    -------
    list[ModelInfo]
        Models whose task type matches ``task_type``.
    """
    return [m for m in models if is_model_compatible_with_task(m, task_type)]
