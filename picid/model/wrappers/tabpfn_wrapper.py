"""Wrapper for the TabPFN forecasting and classification models."""

import torch
from .base import AbstractFeedForwardWrapper

from tabpfn import TabPFNRegressor, TabPFNClassifier
from ...baselines.definitions import CLASSIFICATION_TASKS, FORECASTING_TASKS


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
        # Define supported tasks based on your new categories
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

        # self.train_dataloader = kwargs['train_dataloader']
        # self.val_dataloader = kwargs['val_dataloader']

        self.aux_context = None
        self.y_aux = None

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.task_type in CLASSIFICATION_TASKS:
            return torch.Tensor(self.backbone.predict_proba(X.numpy()))
        else:
            return torch.Tensor(self.backbone.predict(X.numpy()))

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

        # Extent the context with auxiliary features to give sence of seasonality
        self.fit(context_fit, target_fit)

        # Extent the context with auxiliary features to give sence of seasonality
        outputs = self.backbone.predict(context.cpu())

        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {
            "predictions": torch.Tensor(outputs).reshape(target.shape),
            "targets": target,
        }

        return model_out


import numpy as np


def generate_and_fit_polynomial(
    total_points=1580,
    desired_fit_degree=4,
    periodicity=24,
    noise_std=0.05,
    seed=42,
    num_context=20,
    skip=700,
):
    np.random.seed(seed)

    def ground_truth_polynomial(x_scaled):
        # Higher-degree nonlinear structure
        p1 = 0.3 * x_scaled**4
        p2 = -0.2 * x_scaled**5
        p3 = 0.1 * x_scaled**6
        periodic = 0.3 * np.sin(2 * np.pi * x_scaled * periodicity)

        # Combine components
        values = p1 + p2 + p3 + periodic
        values -= values.min()
        values /= values.max()
        return values

    # --- FIX 1: Scale x to [-1, 1] ---
    x_raw = np.linspace(500, 2000, total_points)
    x_vals = np.linspace(
        -1, 1, total_points
    )  # fit over [-1, 1] for numerical stability

    # --- FIX 2: Generate target y ---
    y_clean = ground_truth_polynomial(x_vals)
    y_vals = y_clean + np.random.normal(0, noise_std, size=total_points)

    coeffs_and_errors = []
    y_true_vals = []
    # --- Fit and evaluate ---
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
