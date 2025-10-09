"""Wrapper for the TimePFN model."""

import logging
from pathlib import Path
from typing import override

import joblib
import torch

from TabDPT.src.tabdpt import TabDPTRegressor
from .base import AbstractFitPredictWrapper

logger = logging.getLogger(__name__)


class FitPredictTabDPTWrapper(AbstractFitPredictWrapper):
    """Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(
        self,
        model_cache_path: str = None,
        yield_strategy: bool = False,
        yield_batch_size: int = 128,
        **kwargs,
    ):
        device = "cuda" if kwargs["device"] == "gpu" else "cpu"
        backbone = TabDPTRegressor(
            inf_batch_size=yield_batch_size,
            device=device,
        )

        logger.info(f"Using TabDPT on device: {backbone.device}, kwargs: {kwargs}")

        self.model_cache_path = model_cache_path
        self.device = device

        super().__init__(
            backbone=backbone,
            yield_strategy=yield_strategy,
            yield_batch_size=yield_batch_size,
            **kwargs,
        )

    @override
    def _call_fit(self, X: torch.Tensor, y: torch.Tensor):
        self.backbone.fit(X.numpy(), y.numpy().ravel())

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
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
            model_path = str(Path(".model_cache") / f"{task_id}.joblib")
        else:
            model_path = str(Path(model_path) / model_path / f"{task_id}.joblib")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        # Ensure parent directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Use joblib to persist the estimator which may contain torch objects
        joblib.dump(self.backbone, model_path)
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
            model_path = str(Path(".model_cache") / f"{task_id}.joblib")
        else:
            model_path = str(Path(model_path) / model_path / f"{task_id}.joblib")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.backbone = joblib.load(model_path)
        # self.backbone = load_fitted_tabpfn_model(model_path, device=self.device)
        return model_path

    @property
    @override
    def allows_multi_target(self) -> bool:
        return False
