from pathlib import Path
from typing import Optional, override
import joblib
import torch

from sklearn.ensemble import IsolationForest

# Canonical task type for this wrapper is anomaly_detection (in CLASSIFICATION_TASKS).
from picid.baselines.definitions import CLASSIFICATION_TASKS
from picid.model.wrappers.base import AbstractFitPredictWrapper


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
        self.supported_tasks = CLASSIFICATION_TASKS
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
        Returns 'logits' for a binary classification task.
        Shape: (N_samples, 2)

        Column 0: Score for Class 0 (Normal)
        Column 1: Score for Class 1 (Anomaly)
        """
        # 1. Get raw scores from Isolation Forest
        # Sklearn rule: > 0 is Normal (Inlier), < 0 is Anomaly (Outlier)
        # Therefore, this raw score is already a "Logit for Class 0"
        raw_scores = self.backbone.decision_function(X.numpy())

        # 2. Convert to PyTorch
        score_normal = torch.from_numpy(raw_scores)  # Higher = More Normal
        score_anomaly = torch.from_numpy(-raw_scores)  # Higher = More Anomalous

        # 3. Stack to create a (N, 2) tensor mimicking a classifier's output
        # [ [Prob_Normal, Prob_Anomaly], ... ]
        logits = torch.stack([score_normal, score_anomaly], dim=1)

        return logits.float()

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
