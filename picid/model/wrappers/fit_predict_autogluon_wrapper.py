"""Wrapper for the TimePFN model."""

from pathlib import Path
from typing import override

import joblib

from .base import AbstractFitPredictWrapper
from ..autogluon.autogluon_wrapper import AutoGluonWrapper
from ...baselines.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS


class FitPredictAutoGluonWrapper(AbstractFitPredictWrapper):
    """
    Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.

    Parameters
    ----------
    task_type : str
        Project task identifier.
    model_cache_path : str, optional
        Directory used to cache serialized models.
    **kwargs : Any
        Additional wrapper arguments forwarded to the AutoGluon backbone.
    """

    def __init__(self, task_type: str, model_cache_path: str = None, **kwargs):
        # Define supported tasks based on your new categories
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task {task_type} not supported for FitPredictAutoGluonWrapper."
            )

        self.task_type = task_type
        backbone = AutoGluonWrapper(**kwargs)

        self.model_cache_path = model_cache_path
        super().__init__(backbone=backbone, **kwargs)

    def serialize_model(self, task_id: str | None = None):
        """
        Serialize the model to a file.

        Parameters
        ----------
        task_id : str, optional
            Task identifier used to derive the output filename.

        Returns
        -------
        str
            Path to the serialized model file.
        """
        model_path = self.model_cache_path
        if model_path is None and task_id is not None:
            model_path = str(Path(".model_cache") / f"{task_id}.joblib")
        else:
            model_path = str(Path(model_path) / model_path / f"{task_id}.joblib")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        # Ensure parent directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Use joblib to persist the estimator which may contain torch objects
        joblib.dump(self.backbone, model_path)
        return model_path

    def load_model(self, task_id: str | None = None):
        """
        Load the model from a file.

        Parameters
        ----------
        task_id : str, optional
            Task identifier used to derive the input filename.

        Returns
        -------
        str
            Path to the loaded model file.
        """
        model_path = self.model_cache_path
        if model_path is None and task_id is not None:
            model_path = str(Path(".model_cache") / f"{task_id}.joblib")
        else:
            model_path = str(Path(model_path) / model_path / f"{task_id}.joblib")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.backbone = joblib.load(model_path)
        return model_path

    @property
    @override
    def allows_multi_target(self) -> bool:
        return True
