"""Wrapper for the TimePFN model."""

import logging
from pathlib import Path
from typing import override

import joblib
from tabpfn import TabPFNRegressor

from tabpfn.model.loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
import torch
from .base import AbstractFitPredictWrapper
from carte_ai import CARTERegressor

logger = logging.getLogger(__name__)


class FitPredictCarteWrapper(AbstractFitPredictWrapper):
    """Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(
        self,
        num_model: int,
        disable_pbar: bool,
        random_state: int,
        device: str,
        n_jobs: int,
        pretrained_model_path: str = None,
        model_cache_path: str = None,
        yield_strategy: bool = False,
        yield_batch_size: int = 128,
        **kwargs,
    ):
        device = "cuda" if device == "gpu" else "cpu"

        fixed_params = dict()
        fixed_params["num_model"] = num_model
        fixed_params["disable_pbar"] = disable_pbar
        fixed_params["random_state"] = random_state
        fixed_params["device"] = device
        fixed_params["n_jobs"] = n_jobs
        fixed_params["pretrained_model_path"] = pretrained_model_path

        backbone = CARTERegressor(**fixed_params)  # CARTERegressor for Regression

        logger.info(f"Using Carte on device: {backbone.device}, kwargs: {kwargs}")

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
        self.backbone.fit(X, y)

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.backbone.predict(X)

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
            model_path = str(Path(".model_cache") / f"{task_id}.tabpfn_fit")
        else:
            model_path = str(Path(model_path) / model_path / f"{task_id}.tabpfn_fit")

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
            model_path = str(Path(".model_cache") / f"{task_id}.tabpfn_fit")
        else:
            model_path = str(Path(model_path) / model_path / f"{task_id}.tabpfn_fit")

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
