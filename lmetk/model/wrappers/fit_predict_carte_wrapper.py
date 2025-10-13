"""Wrapper for the TimePFN model."""

import logging
from pathlib import Path
from typing import override

import joblib
import pandas as pd
import torch
from carte_ai import CARTERegressor, Table2GraphTransformer
from carte_ai.configs.directory import config_directory
from huggingface_hub import hf_hub_download

from .base import AbstractFitPredictWrapper

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
        max_epoch: int = 500,
        pretrained_model_path: str = None,
        model_cache_path: str = None,
        yield_strategy: bool = False,
        yield_batch_size: int = 128,
        **kwargs,
    ):
        # Lightning uses 'cuda' or 'gpu' as device string
        device = "cuda" if (device == "gpu") or ("cuda" in device) else "cpu"

        fixed_params = dict()
        fixed_params["num_model"] = num_model
        fixed_params["disable_pbar"] = disable_pbar
        fixed_params["random_state"] = random_state
        fixed_params["device"] = device
        fixed_params["n_jobs"] = n_jobs
        fixed_params["pretrained_model_path"] = config_directory["pretrained_model"]
        fixed_params["max_epoch"] = max_epoch

        backbone = CARTERegressor(**fixed_params)  # CARTERegressor for Regression
        model_path = hf_hub_download(
            repo_id="hi-paris/fastText", filename="cc.en.300.bin"
        )

        logger.info(f"Using Carte on device: {backbone.device}, kwargs: {kwargs}")

        self.model_cache_path = model_cache_path
        self.device = device
        self.backbone = backbone
        self.preprocessor = Table2GraphTransformer(fasttext_model_path=model_path)

        super().__init__(
            backbone=backbone,
            yield_strategy=yield_strategy,
            yield_batch_size=yield_batch_size,
            **kwargs,
        )

    @override
    def _call_fit(self, X: torch.Tensor, y: torch.Tensor):
        X_train = pd.DataFrame(X.numpy())
        y_train = y.squeeze(1).numpy()  # Ensure y is 1
        X_train = self.preprocessor.fit_transform(X_train, y=y_train)
        self.backbone.fit(X=X_train, y=y_train)

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        X_test = pd.DataFrame(X.numpy())
        # Transform the test data
        X_test = self.preprocessor.transform(X_test)
        y_pred = self.backbone.predict(X_test)
        return y_pred

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
