"""Fit/predict wrapper for TabDPT models."""

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
    """
    Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.

    Parameters
    ----------
    device : str
        Device specifier used by TabDPT.
    task_type : str
        Project task identifier.
    model_cache_path : str, optional
        Directory used to cache serialized models.
    yield_strategy : bool, default=False
        Whether batched prediction is enabled.
    yield_batch_size : int, default=128
        Batch size for the batched prediction path.
    **kwargs : Any
        Additional wrapper arguments forwarded to the backbone.
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
        """
        Create the TabDPT wrapper and deferred backbone factory.

        Parameters
        ----------
        device : str
            Device specifier used by TabDPT.
        task_type : str
            Project task identifier.
        model_cache_path : str, optional
            Directory used to cache serialized models.
        yield_strategy : bool, default=False
            Whether batched prediction is enabled.
        yield_batch_size : int, default=128
            Batch size used by the deferred backbone factory.
        **kwargs : Any
            Additional wrapper arguments forwarded to the backbone.
        """
        # Define supported tasks based on your new categories
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task {task_type} not supported for FitPredictTabDPTWrapper."
            )

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
        """
        Recreate the TabDPT backbone when needed and fit it on numpy arrays.

        Parameters
        ----------
        X : torch.Tensor
            Feature tensor used for fitting.
        y : torch.Tensor
            Target tensor aligned with ``X``.
        """
        if self.reinit_on_fit:
            self._reinit_backbone()

        self.backbone.fit(X.numpy(), y.numpy().ravel())

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict regression values or class probabilities with TabDPT.

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
            return torch.Tensor(self.backbone.predict_proba(X.numpy()))
        else:
            return torch.Tensor(self.backbone.predict(X.numpy()))

    @override
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
        model_path = str(model_dir / f"{task_id}.cloudpickle")

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
        model_path = str(model_dir / f"{task_id}.cloudpickle")

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
        """
        Return whether the TabDPT wrapper supports multi-target prediction.

        Returns
        -------
        bool
            ``False`` because this wrapper exposes a single prediction stream.
        """
        return False
