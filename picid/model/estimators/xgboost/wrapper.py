"""Fit/predict wrapper for gradient-boosted sklearn estimators."""

from pathlib import Path
from typing import Optional, override
import joblib
import torch

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from picid.model.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from picid.model.adapters.base import AbstractFitPredictWrapper


class FitPredictXGBoostWrapper(AbstractFitPredictWrapper):
    """
    Wrap gradient-boosted tree baselines for regression and classification.

    Parameters
    ----------
    task_type : str
        Project task identifier.
    n_estimators : int, default=1000
        Number of boosting stages.
    model_cache_path : str, optional
        Directory used to cache serialized models.
    random_state : int, optional
        Random seed for the underlying estimator.
    num_classes : int, optional
        Number of output classes for classification tasks.
    **kwargs : Any
        Additional wrapper arguments forwarded to the backbone.
    """

    def __init__(
        self,
        task_type: str,
        n_estimators: int = 1000,
        model_cache_path: str = None,
        random_state: Optional[int] = 42,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        """
        Create the boosted-tree wrapper for regression or classification tasks.

        Parameters
        ----------
        task_type : str
            Project task identifier.
        n_estimators : int, default=1000
            Number of boosting stages.
        model_cache_path : str, optional
            Directory used to cache serialized models.
        random_state : int, optional
            Random seed for the underlying estimator.
        num_classes : int, optional
            Number of output classes for classification tasks.
        **kwargs : Any
            Additional wrapper arguments retained by the base class.
        """
        # Define supported tasks based on your new categories
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks
        self.task_type = task_type
        self.num_classes = num_classes

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task {task_type} not supported for FitPredictXGBoostWrapper."
            )

        # --- Model Configuration ---
        # 1. Determine the model's task_type and num_classes
        if task_type in self.classification_tasks:
            assert (
                self.num_classes is not None
            ), "num_classes must be provided for classification tasks."

            backbone = GradientBoostingClassifier(
                n_estimators=n_estimators, random_state=random_state, verbose=2
            )
        else:
            backbone = GradientBoostingRegressor(
                n_estimators=n_estimators, random_state=random_state, verbose=2
            )

        self.model_cache_path = model_cache_path
        super().__init__(backbone=backbone, task_type=task_type, **kwargs)

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return regression predictions or class probabilities as tensors.

        Parameters
        ----------
        X : torch.Tensor
            Feature tensor to score.

        Returns
        -------
        torch.Tensor
            Regression predictions or class probabilities.
        """
        if self.task_type in CLASSIFICATION_TASKS:
            result = torch.Tensor(self.backbone.predict_proba(X.numpy()))
            assert (
                result.shape[-1] == self.num_classes
            ), "Predicted probabilities do not match number of expected classes."
            return result
        else:
            return torch.Tensor(self.backbone.predict(X.numpy()))

    @override
    def serialize_model(self, task_id):
        """
        Serialize the model to a file using joblib.

        Parameters
        ----------
        task_id : str
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

    @override
    def load_model(self, task_id):
        """
        Load the model from a file using joblib.

        Parameters
        ----------
        task_id : str
            Task identifier used to derive the input filename.

        Returns
        -------
        object
            Deserialized model object.
        """
        if task_id is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")
        model_dir = (
            Path(".model_cache")
            if self.model_cache_path is None
            else Path(self.model_cache_path)
        )
        model_path = str(model_dir / f"{task_id}.joblib")

        return joblib.load(model_path)

    @property
    @override
    def allows_multi_target(self) -> bool:
        """
        Return whether the model supports multi-target prediction.

        Returns
        -------
        bool
            ``False`` because the wrapper returns a single prediction stream.
        """
        return False


__all__ = ["FitPredictXGBoostWrapper"]
