"""Fit/predict wrapper for AutoGluon-backed tabular models."""

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
        """
        Create the AutoGluon fit/predict wrapper for the selected task.

        Parameters
        ----------
        task_type : str
            Project task identifier.
        model_cache_path : str, optional
            Directory used to cache serialized models.
        **kwargs : Any
            Additional wrapper arguments forwarded to the AutoGluon backbone.
        """
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
        if task_id is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")
        model_dir = (
            Path(".model_cache")
            if self.model_cache_path is None
            else Path(self.model_cache_path)
        )
        model_path = str(model_dir / f"{task_id}.joblib")

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
        if task_id is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")
        model_dir = (
            Path(".model_cache")
            if self.model_cache_path is None
            else Path(self.model_cache_path)
        )
        model_path = str(model_dir / f"{task_id}.joblib")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.backbone = joblib.load(model_path)
        return model_path

    @property
    @override
    def allows_multi_target(self) -> bool:
        """
        Return whether the wrapped AutoGluon model supports multi-target outputs.

        Returns
        -------
        bool
            ``True`` because this wrapper can surface multi-target predictions.
        """
        return True
