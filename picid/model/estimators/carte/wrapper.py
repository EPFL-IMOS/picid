"""Wrapper for the CARTE regression model."""

import logging
from pathlib import Path
from typing import override

import joblib
import pandas as pd
import torch
from carte_ai import CARTERegressor, Table2GraphTransformer
from carte_ai.configs.directory import config_directory
from huggingface_hub import hf_hub_download

from picid.model.adapters.base import AbstractFitPredictWrapper
from picid.model.definitions import CLASSIFICATION_TASKS, REGRESSION_TASKS

logger = logging.getLogger(__name__)


class FitPredictCarteWrapper(AbstractFitPredictWrapper):
    """
    Wrap CARTE in the fit/predict model interface.

    Parameters
    ----------
    num_model : int
        Number of CARTE submodels.
    disable_pbar : bool
        Disable CARTE progress bars.
    random_state : int
        Random seed for CARTE.
    device : str
        Device specifier used by CARTE.
    n_jobs : int
        Parallel worker count.
    task_type : str
        Project task identifier.
    max_epoch : int, default=500
        Maximum CARTE training epochs.
    pretrained_model_path : str, optional
        Optional pretrained CARTE checkpoint path.
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
        num_model: int,
        disable_pbar: bool,
        random_state: int,
        device: str,
        n_jobs: int,
        task_type: str,
        max_epoch: int = 500,
        pretrained_model_path: str = None,
        model_cache_path: str = None,
        yield_strategy: bool = False,
        yield_batch_size: int = 128,
        **kwargs,
    ):
        """
        Create the CARTE wrapper and its preprocessing bridge.

        Parameters
        ----------
        num_model : int
            Number of CARTE submodels.
        disable_pbar : bool
            Whether to disable CARTE progress bars.
        random_state : int
            Random seed for CARTE.
        device : str
            Device specifier used by CARTE.
        n_jobs : int
            Parallel worker count.
        task_type : str
            Project task identifier.
        max_epoch : int, default=500
            Maximum CARTE training epochs.
        pretrained_model_path : str, optional
            Optional pretrained checkpoint path.
        model_cache_path : str, optional
            Directory used to cache serialized models.
        yield_strategy : bool, default=False
            Whether batched prediction is enabled.
        yield_batch_size : int, default=128
            Batch size for the batched prediction path.
        **kwargs : Any
            Additional wrapper arguments forwarded to the backbone.
        """
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task {task_type} not supported for FitPredictCarteWrapper."
            )

        self.task_type = task_type
        device = "cuda" if (device == "gpu") or ("cuda" in device) else "cpu"

        fixed_params = {
            "num_model": num_model,
            "disable_pbar": disable_pbar,
            "random_state": random_state,
            "device": device,
            "n_jobs": n_jobs,
            "pretrained_model_path": config_directory["pretrained_model"],
            "max_epoch": max_epoch,
        }

        backbone = CARTERegressor(**fixed_params)
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
        """
        Convert tensor inputs to tabular form and fit the CARTE backbone.

        Parameters
        ----------
        X : torch.Tensor
            Feature tensor to fit on.
        y : torch.Tensor
            Target tensor aligned with ``X``.
        """
        x_train = pd.DataFrame(X.numpy())
        y_train = y.squeeze(1).numpy()
        x_train = self.preprocessor.fit_transform(x_train, y=y_train)
        self.backbone.fit(X=x_train, y=y_train)

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform tensor features and delegate prediction to CARTE.

        Parameters
        ----------
        X : torch.Tensor
            Feature tensor to score.

        Returns
        -------
        torch.Tensor
            Prediction tensor returned by the wrapped CARTE model.
        """
        x_test = pd.DataFrame(X.numpy())
        x_test = self.preprocessor.transform(x_test)
        y_pred = self.backbone.predict(x_test)
        return y_pred

    @override
    def serialize_model(self, task_id: str | None = None):
        """
        Serialize the fitted CARTE model to disk.

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
        model_path = str(model_dir / f"{task_id}.tabpfn_fit")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.backbone, model_path)
        return model_path

    @override
    def load_model(self, task_id: str | None = None):
        """
        Load a previously serialized CARTE model from disk.

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
        model_path = str(model_dir / f"{task_id}.tabpfn_fit")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.backbone = joblib.load(model_path)
        return model_path

    @property
    @override
    def allows_multi_target(self) -> bool:
        """
        Return whether the CARTE wrapper supports multi-target prediction.

        Returns
        -------
        bool
            ``False`` because this wrapper exposes a single prediction stream.
        """
        return False


__all__ = ["FitPredictCarteWrapper"]
