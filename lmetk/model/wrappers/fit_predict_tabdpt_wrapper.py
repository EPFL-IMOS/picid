"""Wrapper for the TimePFN model."""

from functools import partial
import logging
from pathlib import Path
from typing import override

import torch
import cloudpickle

from tabdpt import TabDPTRegressor, TabDPTClassifier
from .base import AbstractFitPredictWrapper
from ...baselines.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS

logger = logging.getLogger(__name__)


class FitPredictTabDPTWrapper(AbstractFitPredictWrapper):
    """Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(
        self,
        device: str,
        task_type: str,
        model_cache_path: str = None,
        yield_strategy: bool = False,
        yield_batch_size: int = 128,
        **kwargs,
    ):

        # Define supported tasks based on your new categories
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(f"Task {task_type} not supported for FitPredictTabDPTWrapper.")

        self.task_type = task_type

        device = "cuda" if (device == "gpu") or ("cuda" in device) else "cpu"
        logger.info(f"Using TabDPT on device: {device}, kwargs: {kwargs}")

        if task_type in self.classification_tasks:
            backbone = partial(
                TabDPTClassifier,
                inf_batch_size=yield_batch_size,
                device=device,
            )
        else:
            backbone = partial(
                TabDPTRegressor,
                inf_batch_size=yield_batch_size,
                device=device,
            )

        self.model_cache_path = model_cache_path
        self.device = device

        super().__init__(
            backbone=backbone,
            yield_strategy=yield_strategy,
            yield_batch_size=yield_batch_size,
            reinit_on_fit=True,
            **kwargs,
        )

    @override
    def _call_fit(self, X: torch.Tensor, y: torch.Tensor):

        if self.reinit_on_fit:
            self._reinit_backbone()

        self.backbone.fit(X.numpy(), y.numpy().ravel())

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.task_type in CLASSIFICATION_TASKS:
            return torch.Tensor(self.backbone.predict_proba(X.numpy()))
        else:
            return torch.Tensor(self.backbone.predict(X.numpy()))

    @override
    def serialize_model(self, task_id: str | None = None):
        """Serialize the model to a file.

        The method prefers `self.kwargs['model_path']`. If that's not set and
        `task_id` is provided, it will write to `models/{task_id}.joblib`.
        Uses joblib.dump to persist the sklearn-style estimator (TabPFNRegressor).
        Returns the path written to.
        """
        model_path = self.model_cache_path
        if model_path is None and task_id is not None:
            model_path = str(Path(".model_cache") / f"{task_id}.cloudpickle")
        else:
            model_path = str(Path(model_path) / model_path / f"{task_id}.cloudpickle")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        # Ensure parent directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Use joblib to persist the estimator which may contain torch objects

        with open(model_path, "wb") as f:
            cloudpickle.dump(self.backbone, f)
        # cloudpickle.dump(self.backbone, model_path)
        # save_fitted_tabpfn_model(self.backbone, model_path)
        return model_path

    @override
    def load_model(self, task_id: str | None = None):
        """Load the model from a file.

        The method prefers `self.kwargs['model_path']`. If that's not set and
        `task_id` is provided, it will attempt to load `models/{task_id}.joblib`.
        Uses joblib.load and replaces `self.backbone` with the loaded estimator.
        Returns the path loaded from.
        """
        model_path = self.model_cache_path
        if model_path is None and task_id is not None:
            model_path = str(Path(".model_cache") / f"{task_id}.cloudpickle")
        else:
            model_path = str(Path(model_path) / model_path / f"{task_id}.cloudpickle")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # self.backbone = cloudpickle.load(model_path)
        with open(model_path, "rb") as f:
            self.backbone = cloudpickle.load(f)
        # self.backbone = load_fitted_tabpfn_model(model_path, device=self.device)
        return model_path

    @property
    @override
    def allows_multi_target(self) -> bool:
        return False
