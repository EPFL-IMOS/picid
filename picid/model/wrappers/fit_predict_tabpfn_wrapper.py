"""Fit/predict wrapper for TabPFN models."""

import logging
from pathlib import Path
from typing import Literal, override

import joblib
import torch

from tabpfn import TabPFNRegressor, TabPFNClassifier

from .base import AbstractFitPredictWrapper
from picid.model.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS

logger = logging.getLogger(__name__)


class FitPredictTabPFNWrapper(AbstractFitPredictWrapper):
    """
    Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.

    Parameters
    ----------
    device : str
        Device specifier used by TabPFN.
    task_type : str
        Project task identifier.
    model_cache_path : str, optional
        Directory used to cache serialized models.
    yield_strategy : bool, default=False
        Whether batched prediction is enabled.
    yield_batch_size : int, default=128
        Batch size for the batched prediction path.
    output_type : {"mean", "full"}, default="mean"
        Output type requested from the regressor when predicting.
    full_outputs_path : str, optional
        Directory used to persist full TabPFN outputs.
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
        output_type: Literal["mean", "full"] = "mean",
        full_outputs_path: str = None,
        **kwargs,
    ):
        """
        Create the TabPFN fit/predict wrapper for the selected task.

        Parameters
        ----------
        device : str
            Device specifier used by TabPFN.
        task_type : str
            Project task identifier.
        model_cache_path : str, optional
            Directory used to cache serialized models.
        yield_strategy : bool, default=False
            Whether batched prediction is enabled.
        yield_batch_size : int, default=128
            Batch size for the batched prediction path.
        output_type : {"mean", "full"}, default="mean"
            Output type requested from the regressor.
        full_outputs_path : str, optional
            Directory used to persist full TabPFN outputs.
        **kwargs : Any
            Additional wrapper arguments forwarded to the backbone.
        """
        # Define supported tasks based on your new categories
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task {task_type} not supported for FitPredictTabPFNWrapper."
            )

        self.task_type = task_type

        device = "cuda" if (device == "gpu") or ("cuda" in device) else "cpu"

        if task_type in self.classification_tasks:
            backbone = TabPFNClassifier(
                ignore_pretraining_limits=kwargs["ignore_pretraining_limits"],
                random_state=kwargs["random_state"],
                device=device,
                n_jobs=kwargs.get("n_jobs", -1),
                fit_mode=kwargs.get("fit_mode", "fit_preprocessors"),
            )
        else:
            backbone = TabPFNRegressor(
                ignore_pretraining_limits=kwargs["ignore_pretraining_limits"],
                random_state=kwargs["random_state"],
                device=device,
                n_jobs=kwargs.get("n_jobs", -1),
                fit_mode=kwargs.get("fit_mode", "fit_preprocessors"),
            )

        logger.info(f"Using TabPFN on device: {backbone.device}, kwargs: {kwargs}")

        self.output_type = output_type
        self.model_cache_path = model_cache_path
        self.device = device
        self.full_outputs_path = full_outputs_path

        super().__init__(
            backbone=backbone,
            yield_strategy=yield_strategy,
            yield_batch_size=yield_batch_size,
            **kwargs,
        )

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities or regression outputs with TabPFN.

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
            if self.full_outputs_path is not None:
                output = self.backbone.predict(X, output_type=self.output_type)
                output["criterion"] = output["criterion"].cpu()
                output["logits"] = output["logits"].cpu()
                output_file = Path(self.full_outputs_path) / "predictions.pt"
                torch.save(output, output_file)
                if self.output_type == "full":
                    output = output["mean"]
                return torch.Tensor(output)
            else:
                return torch.Tensor(self.backbone.predict(X.numpy()))

    # @override
    # def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
    #     """Inner predict call to be potentially overridden by subclasses."""
    #     output = self.backbone.predict(X, output_type=self.output_type)
    #     if self.full_outputs_path is not None:
    #         output["criterion"] = output["criterion"].cpu()
    #         output["logits"] = output["logits"].cpu()
    #         output_file = Path(self.full_outputs_path) / "predictions.pt"
    #         torch.save(output, output_file)
    #         if self.output_type == "full":
    #             output = output["mean"]
    #     return output

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
        model_path = str(model_dir / f"{task_id}.tabpfn_fit")

        # Ensure parent directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Use joblib to persist the estimator which may contain torch objects
        joblib.dump(self.backbone, model_path)
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
        model_path = str(model_dir / f"{task_id}.tabpfn_fit")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.backbone = joblib.load(model_path)
        # self.backbone = load_fitted_tabpfn_model(model_path, device=self.device)
        return model_path

    @property
    @override
    def allows_multi_target(self) -> bool:
        """
        Return whether the TabPFN fit/predict wrapper supports multi-target outputs.

        Returns
        -------
        bool
            ``False`` because this wrapper exposes a single prediction stream.
        """
        return False
