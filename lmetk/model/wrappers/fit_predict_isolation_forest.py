from pathlib import Path
from typing import Optional, override
import joblib
import torch
import numpy as np

from sklearn.ensemble import IsolationForest

# Assuming "detection" is the task type you are using,
# you might import DETECTION_TASKS or just define it here as a supported type.
from lmetk.baselines.definitions import CLASSIFICATION_TASKS
from lmetk.model.wrappers.base import AbstractFitPredictWrapper

# If DETECTION_TASKS is not in your definitions file, we define it locally for the check
DETECTION_TASKS = ["detection", "anomaly_detection"]


class FitPredictIsolationForestWrapper(AbstractFitPredictWrapper):
    """Wrapper for the Isolation Forest model for anomaly detection."""

    def __init__(
        self,
        task_type: str,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        model_cache_path: str = None,
        random_state: Optional[int] = 42,
        **kwargs,
    ):

        # --- Task Validation ---
        self.supported_tasks = DETECTION_TASKS + CLASSIFICATION_TASKS
        self.task_type = task_type

        if task_type not in self.supported_tasks:
            raise ValueError(
                f"Task {task_type} not supported for FitPredictIsolationForestWrapper. "
                f"Supported tasks: {self.supported_tasks}"
            )

        # --- Model Configuration ---
        # IsolationForest is unsupervised, so we don't need num_classes logic here.
        backbone = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,  # Parallelize by default for speed
            verbose=0,
        )

        self.model_cache_path = model_cache_path
        super().__init__(backbone=backbone, task_type=task_type, **kwargs)

    @override
    def _call_fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Override fit to explicitly ignore target labels 'y',
        as Isolation Forest is unsupervised.
        """
        if self.reinit_on_fit:
            self._reinit_backbone()

        # IsolationForest.fit(X, y=None, sample_weight=None)
        # We pass only X (converted to numpy)
        self.backbone.fit(X.numpy())

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns anomaly scores.

        Standard IsolationForest.decision_function output:
           > 0 : Inlier (Normal)
           < 0 : Outlier (Anomaly)

        We typically negate this for Anomaly Detection tasks so that:
           Higher Score = More Anomalous
        """
        # Get decision function scores
        scores = self.backbone.decision_function(X.numpy())

        # Invert sign so higher value == higher anomaly score
        # (Standard practice for AUC calculation in anomaly detection)
        anomaly_scores = -scores

        return torch.Tensor(anomaly_scores)

    @override
    def serialize_model(self, task_id):
        """
        Serialize the model to a file using joblib.
        Matches the logic in FitPredictXGBoostWrapper.
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

        # Use joblib to persist the estimator
        joblib.dump(self.backbone, model_path)
        return model_path

    @override
    def load_model(self, task_id):
        """
        Load the model from a file using joblib.
        Matches the logic in FitPredictXGBoostWrapper.
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
