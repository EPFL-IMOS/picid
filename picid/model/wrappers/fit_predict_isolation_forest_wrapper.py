"""Fit/predict wrapper for anomaly scoring with Isolation Forest."""

from pathlib import Path
from typing import Optional, override
import joblib
import torch

from sklearn.ensemble import IsolationForest

# Canonical task type for this wrapper is anomaly_detection (in CLASSIFICATION_TASKS).
from picid.model.definitions import CLASSIFICATION_TASKS
from picid.model.wrappers.base import AbstractFitPredictWrapper


class FitPredictIsolationForestWrapper(AbstractFitPredictWrapper):
    """
    Wrap Isolation Forest in the fit/predict model interface.

    Parameters
    ----------
    task_type : str
        Project task identifier.
    n_estimators : int, default=100
        Number of trees in the forest.
    contamination : float or str, default="auto"
        Expected contamination level passed to scikit-learn.
    model_cache_path : str, optional
        Directory used to cache serialized models.
    random_state : int, optional
        Random seed for scikit-learn.
    **kwargs : Any
        Additional wrapper arguments forwarded to the backbone.
    """

    def __init__(
        self,
        task_type: str,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        model_cache_path: str = None,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        """
        Create the Isolation Forest wrapper for anomaly-detection tasks.

        Parameters
        ----------
        task_type : str
            Project task identifier.
        n_estimators : int, default=100
            Number of trees in the forest.
        contamination : float or str, default="auto"
            Contamination passed to the sklearn backbone.
        model_cache_path : str, optional
            Directory used to cache serialized models.
        random_state : int, optional
            Random seed for sklearn.
        **kwargs : Any
            Additional wrapper arguments retained by the base class.
        """
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
        Fit Isolation Forest while ignoring target labels.

        Parameters
        ----------
        X : torch.Tensor
            Input batch in tensor form.
        y : torch.Tensor
            Target batch in tensor form. The values are ignored.
        """
        if self.reinit_on_fit:
            self._reinit_backbone()

        # IsolationForest.fit(X, y=None, sample_weight=None)
        # We pass only X (converted to numpy)
        self.backbone.fit(X.numpy())

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return anomaly scores shaped like binary classification logits.

        Parameters
        ----------
        X : torch.Tensor
            Input batch in tensor form.

        Returns
        -------
        torch.Tensor
            Two-column tensor of normal/anomaly scores.
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
        Serialize the fitted Isolation Forest model to disk.

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

        # Use joblib to persist the estimator
        joblib.dump(self.backbone, model_path)
        return model_path

    @override
    def load_model(self, task_id):
        """
        Load a previously serialized Isolation Forest model.

        Parameters
        ----------
        task_id : str
            Task identifier used to derive the input filename.

        Returns
        -------
        object
            Deserialized Isolation Forest estimator.
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
            ``False`` because Isolation Forest only predicts a single target
            stream in this wrapper.
        """
        return False
