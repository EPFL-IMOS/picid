"""Canonical TabPFN estimator wrappers."""

import logging
from pathlib import Path
from typing import Literal, override

import joblib
import numpy as np
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor

from picid.model.adapters.base import (
    AbstractFeedForwardWrapper,
    AbstractFitPredictWrapper,
)
from picid.model.definitions import CLASSIFICATION_TASKS, FORECASTING_TASKS
from picid.model.definitions import REGRESSION_TASKS

logger = logging.getLogger(__name__)


class FitPredictTabPFNWrapper(AbstractFitPredictWrapper):
    """
    Wrap TabPFN models in the fit/predict model interface.

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
        if self.full_outputs_path is not None:
            output = self.backbone.predict(X, output_type=self.output_type)
            output["criterion"] = output["criterion"].cpu()
            output["logits"] = output["logits"].cpu()
            output_file = Path(self.full_outputs_path) / "predictions.pt"
            torch.save(output, output_file)
            if self.output_type == "full":
                output = output["mean"]
            return torch.Tensor(output)
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
        model_path = str(model_dir / f"{task_id}.tabpfn_fit")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.backbone, model_path)
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


class TabPFNWrapper(AbstractFeedForwardWrapper):
    """
    Wrap TabPFN in the project model interface.

    Parameters
    ----------
    task_type : str
        Project task identifier.
    **kwargs : Any
        Additional wrapper arguments forwarded to the backbone.
    """

    def __init__(self, task_type: str, **kwargs):
        """
        Create the wrapper and choose the TabPFN backbone for the task.

        Parameters
        ----------
        task_type : str
            Project task identifier.
        **kwargs : Any
            Additional arguments forwarded to the TabPFN constructor.
        """
        self.regression_tasks = FORECASTING_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(f"Task {task_type} not supported for TabPFNWrapper.")

        self.task_type = task_type

        if task_type in self.classification_tasks:
            backbone = TabPFNClassifier(
                ignore_pretraining_limits=kwargs["ignore_pretraining_limits"],
                random_state=kwargs["random_state"],
                device="cuda" if kwargs["device"] == "gpu" else "cpu",
            )
        else:
            backbone = TabPFNRegressor(
                ignore_pretraining_limits=kwargs["ignore_pretraining_limits"],
                random_state=kwargs["random_state"],
                device="cuda" if kwargs["device"] == "gpu" else "cpu",
            )
        super().__init__(backbone=backbone, **kwargs)
        self.model_fit_flag = False
        self.lag = 0
        self.aux_context = None
        self.y_aux = None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the TabPFN backbone from torch tensors.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix used to fit the wrapped backbone.
        y : torch.Tensor
            Target matrix paired with ``X`` during fitting.

        Returns
        -------
        TabPFNWrapper
            The current wrapper instance.
        """
        self.backbone.fit(
            X.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
        )
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict from a torch tensor using the wrapped TabPFN backbone.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix to score.

        Returns
        -------
        torch.Tensor
            Model predictions or class probabilities.
        """
        x_numpy = X.detach().cpu().numpy()
        if self.task_type in CLASSIFICATION_TASKS:
            return torch.Tensor(self.backbone.predict_proba(x_numpy))
        return torch.Tensor(self.backbone.predict(x_numpy))

    def forward(self, batch):
        """
        Run the TabPFN wrapper on one batch.

        Parameters
        ----------
        batch : dict
            Batch dictionary containing context and target tensors.

        Returns
        -------
        dict
            Predictions and targets for the training loop.
        """
        context_fit = batch["context"]["x"].squeeze(0)
        target_fit = batch["target"]["x"].squeeze(0)

        context = batch["context"]["y"].squeeze(0)
        target = batch["target"]["y"].squeeze(0)

        self.fit(context_fit, target_fit)
        outputs = self.backbone.predict(context.cpu())

        model_out = {
            "predictions": torch.Tensor(outputs).reshape(target.shape),
            "targets": target,
        }
        return model_out


def generate_and_fit_polynomial(
    total_points=1580,
    desired_fit_degree=4,
    periodicity=24,
    noise_std=0.05,
    seed=42,
    num_context=20,
    skip=700,
):
    """
    Generate a synthetic polynomial series and fit rolling polynomial forecasts.

    Parameters
    ----------
    total_points : int, default=1580
        Number of points in the generated series.
    desired_fit_degree : int, default=4
        Polynomial degree used for the rolling fit.
    periodicity : int, default=24
        Frequency multiplier for the sinusoidal component.
    noise_std : float, default=0.05
        Standard deviation of the Gaussian noise added to the signal.
    seed : int, default=42
        Random seed for reproducible noise generation.
    num_context : int, default=20
        Number of rolling context windows to evaluate.
    skip : int, default=700
        Starting index for the first rolling fit.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Raw x positions, noisy series values, fitted coefficients with errors,
        and the held-out true values.
    """
    np.random.seed(seed)

    def ground_truth_polynomial(x_scaled):
        """
        Construct the normalized nonlinear signal used for the toy series.

        Parameters
        ----------
        x_scaled : np.ndarray
            Input coordinates scaled to ``[-1, 1]``.

        Returns
        -------
        np.ndarray
            Normalized nonlinear signal evaluated at ``x_scaled``.
        """
        p1 = 0.3 * x_scaled**4
        p2 = -0.2 * x_scaled**5
        p3 = 0.1 * x_scaled**6
        periodic = 0.3 * np.sin(2 * np.pi * x_scaled * periodicity)

        values = p1 + p2 + p3 + periodic
        values -= values.min()
        values /= values.max()
        return values

    x_raw = np.linspace(500, 2000, total_points)
    x_vals = np.linspace(-1, 1, total_points)

    y_clean = ground_truth_polynomial(x_vals)
    y_vals = y_clean + np.random.normal(0, noise_std, size=total_points)

    coeffs_and_errors = []
    y_true_vals = []

    for i in range(skip, skip + num_context, 1):
        x_obs = x_vals[:i]
        y_obs = y_vals[:i]

        coeffs = np.polyfit(x_obs, y_obs, desired_fit_degree)
        poly = np.poly1d(coeffs)

        x_next = x_vals[i]
        y_pred = poly(x_next)
        y_true = y_vals[i]

        error = y_pred - y_true
        coeffs_and_errors.append(np.concatenate([coeffs, [error]]))
        y_true_vals.append(y_true)

    return x_raw, y_vals, np.array(coeffs_and_errors), np.array(y_true_vals)


__all__ = ["FitPredictTabPFNWrapper", "TabPFNWrapper", "generate_and_fit_polynomial"]
