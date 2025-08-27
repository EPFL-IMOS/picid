"""Wrapper for the TimePFN model."""

from pathlib import Path

import joblib
from autogluon.tabular import TabularPredictor

from .base import AbstractFitPredictWrapper
from ..autogluon.autogluon_wrapper import AutoGluonWrapper


class FitPredictAutoGluonWrapper(AbstractFitPredictWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(self, model_cache_path: str = None, **kwargs):
        backbone = AutoGluonWrapper(**kwargs)

        self.model_cache_path = model_cache_path
        super().__init__(backbone=backbone, **kwargs)

    def serialize_model(self, task_id: str | None = None):
        """Serialize the model to a file.

        The method prefers `self.kwargs['model_path']`. If that's not set and
        `task_id` is provided, it will write to `models/{task_id}.joblib`.
        Uses joblib.dump to persist the sklearn-style estimator (TabPFNRegressor).
        Returns the path written to.
        """
        model_path = self.model_cache_path
        if model_path is None and task_id is not None:
            model_path = str(Path(".model_cache") / f"{task_id}.joblib")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        # Ensure parent directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Use joblib to persist the estimator which may contain torch objects
        joblib.dump(self.backbone, model_path)
        return model_path

    def load_model(self, task_id: str | None = None):
        """Load the model from a file.

        The method prefers `self.kwargs['model_path']`. If that's not set and
        `task_id` is provided, it will attempt to load `models/{task_id}.joblib`.
        Uses joblib.load and replaces `self.backbone` with the loaded estimator.
        Returns the path loaded from.
        """
        model_path = self.model_cache_path
        if model_path is None and task_id is not None:
            model_path = str(Path(".model_cache") / f"{task_id}.joblib")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.backbone = joblib.load(model_path)
        return model_path
