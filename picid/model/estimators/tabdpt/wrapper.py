"""Fit/predict wrapper for TabDPT models."""

from functools import partial
import logging
from pathlib import Path
from typing import override

import cloudpickle
import torch
from tabdpt import TabDPTClassifier, TabDPTRegressor

from picid.model.adapters.base import AbstractFitPredictWrapper
from picid.model.definitions import CLASSIFICATION_TASKS, REGRESSION_TASKS

logger = logging.getLogger(__name__)


class FitPredictTabDPTWrapper(AbstractFitPredictWrapper):
    """
    Wrap TabDPT models in the fit/predict model interface.

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
    random_state : int or None, default=None
        Seed for the regression prediction ensemble. ``None`` retains unseeded
        prediction. Classification does not use this setting.
    **kwargs : Any
        Additional wrapper metadata stored by the base adapter.
    """

    def __init__(
        self,
        device: str,
        task_type: str,
        model_cache_path: str = None,
        yield_strategy: bool = False,
        yield_batch_size: int = 128,
        random_state: int | None = None,
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
        random_state : int or None, default=None
            Regression ensemble seed, reused for each prediction call and batch.
            ``None`` preserves the unseeded behavior. Classification is unchanged.
        **kwargs : Any
            Additional wrapper metadata stored by the base adapter.
        """
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task {task_type} not supported for FitPredictTabDPTWrapper."
            )

        self.task_type = task_type
        self.random_state = random_state
        self._prediction_seed = random_state

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
            random_state=random_state,
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
        self._prediction_seed = self.random_state

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
            return torch.Tensor(
                self.backbone.predict(
                    X.numpy(), n_ensembles=8, seed=self._prediction_seed
                )
            )

    @override
    def serialize_model(self, task_id: str | None = None):
        """
        Serialize the model and its effective regression prediction seed.

        Classification checkpoints retain the backbone-only format.

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

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self.backbone
        if self.task_type in REGRESSION_TASKS:
            checkpoint = {
                "format": "picid.tabdpt",
                "version": 1,
                "backbone": self.backbone,
                "prediction_seed": self._prediction_seed,
            }
        with open(model_path, "wb") as f:
            cloudpickle.dump(checkpoint, f)
        return model_path

    @override
    def load_model(self, task_id: str | None = None):
        """
        Load the model and restore its regression prediction seed.

        Legacy backbone-only checkpoints use the configured ``random_state``.
        They do not contain the internal seeds used for historical predictions.
        Fitting again uses the configured seed rather than the loaded seed.

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

        with open(model_path, "rb") as f:
            checkpoint = cloudpickle.load(f)
        if isinstance(checkpoint, dict) and checkpoint.get("format") == "picid.tabdpt":
            if checkpoint.get("version") != 1:
                raise ValueError(
                    f"Unsupported TabDPT checkpoint version: {checkpoint.get('version')}"
                )
            backbone = checkpoint["backbone"]
            prediction_seed = checkpoint["prediction_seed"]
        else:
            backbone = checkpoint
            prediction_seed = self.random_state
        self.backbone = backbone
        self._prediction_seed = prediction_seed
        if self.task_type in REGRESSION_TASKS:
            logger.info("Loaded TabDPT regression prediction seed: %s", prediction_seed)
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


__all__ = ["FitPredictTabDPTWrapper"]
