from pathlib import Path
from typing import Optional, override
import joblib
import torch

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

from lmetk.baselines.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from lmetk.model.wrappers.base import AbstractFitPredictWrapper


class FitPredictXGBoostWrapper(AbstractFitPredictWrapper):
    """Wrapper for the XGboost model for regression and classification."""

    def __init__(
        self,
        task_type: str,
        n_estimators: int = 100,
        model_cache_path: str = None,
        random_state: Optional[int] = 42,
        **kwargs,
    ):

        # Define supported tasks based on your new categories
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks
        self.task_type = task_type

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task {task_type} not supported for FitPredictXGBoostWrapper."
            )

        # --- Model Configuration ---
        # 1. Determine the model's task_type and num_classes
        if task_type in self.classification_tasks:
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
        if self.task_type in CLASSIFICATION_TASKS:
            return torch.Tensor(self.backbone.predict_proba(X.numpy()))
        else:
            return torch.Tensor(self.backbone.predict(X.numpy()))

    @override
    def serialize_model(self, task_id):
        """
        Serialize the model to a file using joblib.
        depending on task_id and self.kwargs['model_path'].
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

    @override
    def load_model(self, task_id):
        """
        Load the model from a file using joblib.
        """
        model_path = self.model_cache_path
        if model_path is None and task_id is not None:
            model_path = str(Path(".model_cache") / f"{task_id}.joblib")
        else:
            model_path = str(Path(model_path) / model_path / f"{task_id}.joblib")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        return joblib.load(model_path)

    @property
    @override
    def allows_multi_target(self) -> bool:
        """Indicates whether the model supports multi-target prediction."""
        return False
