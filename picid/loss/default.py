"""
Default loss implementations for time series transformer models.

This module provides concrete implementations of common loss functions
used in time series forecasting and analysis tasks.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from .base import AbstractLoss


class MSELoss(AbstractLoss):
    """
    Mean squared error loss for time-series regression.

    Parameters
    ----------
    reduction : str, default="mean"
        Reduction passed to :class:`torch.nn.MSELoss`.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Initialize the loss wrapper.

        Parameters
        ----------
        reduction : str, default="mean"
            Reduction passed to :class:`torch.nn.MSELoss`.
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the batch loss and return it inside ``model_out``.

        Parameters
        ----------
        model_out : dict[str, torch.Tensor]
            Model outputs containing the ``predictions`` tensor.
        batch : dict[str, torch.Tensor]
            Batch data containing the ``targets`` tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Copy of ``model_out`` with an added ``loss`` entry.
        """
        predictions = model_out["predictions"]
        targets = model_out["targets"]

        loss = self.mse_loss(predictions, targets)

        result = model_out.copy()
        result["loss"] = loss
        return result


class MAELoss(AbstractLoss):
    """
    Mean absolute error loss for time-series regression.

    Parameters
    ----------
    reduction : str, default="mean"
        Reduction passed to :class:`torch.nn.L1Loss`.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Initialize the loss wrapper.

        Parameters
        ----------
        reduction : str, default="mean"
            Reduction passed to :class:`torch.nn.L1Loss`.
        """
        super().__init__()
        self.mae_loss = nn.L1Loss(reduction=reduction)

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the batch loss and return it inside ``model_out``.

        Parameters
        ----------
        model_out : dict[str, torch.Tensor]
            Model outputs containing the ``predictions`` tensor.
        batch : dict[str, torch.Tensor]
            Batch data containing the ``targets`` tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Copy of ``model_out`` with an added ``loss`` entry.
        """
        predictions = model_out["predictions"]
        targets = batch["targets"]

        loss = self.mae_loss(predictions, targets)

        result = model_out.copy()
        result["loss"] = loss
        return result


class HuberLoss(AbstractLoss):
    """
    Huber loss for time-series regression.

    Parameters
    ----------
    delta : float, default=1.0
        Threshold that switches between quadratic and linear penalties.
    reduction : str, default="mean"
        Reduction passed to :class:`torch.nn.HuberLoss`.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        """
        Initialize the loss wrapper.

        Parameters
        ----------
        delta : float, default=1.0
            Threshold that switches between quadratic and linear penalties.
        reduction : str, default="mean"
            Reduction passed to :class:`torch.nn.HuberLoss`.
        """
        super().__init__()
        self.huber_loss = nn.HuberLoss(delta=delta, reduction=reduction)

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the batch loss and return it inside ``model_out``.

        Parameters
        ----------
        model_out : dict[str, torch.Tensor]
            Model outputs containing the ``predictions`` tensor.
        batch : dict[str, torch.Tensor]
            Batch data containing the ``targets`` tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Copy of ``model_out`` with an added ``loss`` entry.
        """
        predictions = model_out["predictions"]
        targets = batch["targets"]

        loss = self.huber_loss(predictions, targets)

        result = model_out.copy()
        result["loss"] = loss
        return result


class QuantileLoss(AbstractLoss):
    """
    Quantile loss for probabilistic forecasting.

    Parameters
    ----------
    quantile : float, default=0.5
        Target quantile in the open interval ``(0, 1)``.
    reduction : str, default="mean"
        Reduction to apply after computing pointwise quantile penalties.
    """

    def __init__(self, quantile: float = 0.5, reduction: str = "mean") -> None:
        """
        Initialize the loss wrapper.

        Parameters
        ----------
        quantile : float, default=0.5
            Target quantile in the open interval ``(0, 1)``.
        reduction : str, default="mean"
            Reduction to apply after computing pointwise quantile penalties.
        """
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction

        if not 0 < quantile < 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the batch loss and return it inside ``model_out``.

        Parameters
        ----------
        model_out : dict[str, torch.Tensor]
            Model outputs containing the ``predictions`` tensor.
        batch : dict[str, torch.Tensor]
            Batch data containing the ``targets`` tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Copy of ``model_out`` with an added ``loss`` entry.
        """
        predictions = model_out["predictions"]
        targets = batch["targets"]

        errors = targets - predictions
        loss = torch.maximum(self.quantile * errors, (self.quantile - 1) * errors)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        result = model_out.copy()
        result["loss"] = loss
        return result


class MAPELoss(AbstractLoss):
    """
    Mean absolute percentage error loss.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Small denominator floor used to avoid division by zero.
    reduction : str, default="mean"
        Reduction to apply after computing pointwise percentage errors.
    """

    def __init__(self, epsilon: float = 1e-8, reduction: str = "mean") -> None:
        """
        Initialize the loss wrapper.

        Parameters
        ----------
        epsilon : float, default=1e-8
            Small denominator floor used to avoid division by zero.
        reduction : str, default="mean"
            Reduction to apply after computing pointwise percentage errors.
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the batch loss and return it inside ``model_out``.

        Parameters
        ----------
        model_out : dict[str, torch.Tensor]
            Model outputs containing the ``predictions`` tensor.
        batch : dict[str, torch.Tensor]
            Batch data containing the ``targets`` tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Copy of ``model_out`` with an added ``loss`` entry.
        """
        predictions = model_out["predictions"]
        targets = batch["targets"]

        # Avoid division by zero
        targets_safe = torch.where(
            torch.abs(targets) < self.epsilon,
            torch.sign(targets) * self.epsilon,
            targets,
        )

        loss = torch.abs((targets - predictions) / targets_safe)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        result = model_out.copy()
        result["loss"] = loss
        return result


class SMAPELoss(AbstractLoss):
    """
    Symmetric mean absolute percentage error loss.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Small denominator floor used to avoid division by zero.
    reduction : str, default="mean"
        Reduction to apply after computing pointwise symmetric percentage errors.
    """

    def __init__(self, epsilon: float = 1e-8, reduction: str = "mean") -> None:
        """
        Initialize the loss wrapper.

        Parameters
        ----------
        epsilon : float, default=1e-8
            Small denominator floor used to avoid division by zero.
        reduction : str, default="mean"
            Reduction to apply after computing pointwise symmetric percentage errors.
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the batch loss and return it inside ``model_out``.

        Parameters
        ----------
        model_out : dict[str, torch.Tensor]
            Model outputs containing the ``predictions`` tensor.
        batch : dict[str, torch.Tensor]
            Batch data containing the ``targets`` tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Copy of ``model_out`` with an added ``loss`` entry.
        """
        predictions = model_out["predictions"]
        targets = batch["targets"]

        numerator = torch.abs(targets - predictions)
        denominator = (torch.abs(targets) + torch.abs(predictions)) / 2 + self.epsilon

        loss = numerator / denominator

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        result = model_out.copy()
        result["loss"] = loss
        return result


class WeightedMSELoss(AbstractLoss):
    """
    Weighted Mean Squared Error loss for time series with sample weights.

    Useful when certain time points or samples should have more importance
    in the loss computation, such as recent observations or specific events.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Initialize the loss wrapper.

        Parameters
        ----------
        reduction : str, default="mean"
            Reduction to apply after weighting squared errors.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the batch loss and return it inside ``model_out``.

        Parameters
        ----------
        model_out : dict[str, torch.Tensor]
            Model outputs containing the ``predictions`` tensor.
        batch : dict[str, torch.Tensor]
            Batch data containing ``targets`` and optionally ``weights``.

        Returns
        -------
        dict[str, torch.Tensor]
            Copy of ``model_out`` with an added ``loss`` entry.
        """
        predictions = model_out["predictions"]
        targets = batch["targets"]
        weights = batch.get("weights", torch.ones_like(targets))

        squared_errors = (predictions - targets) ** 2
        weighted_errors = squared_errors * weights

        if self.reduction == "mean":
            loss = weighted_errors.sum() / weights.sum()
        elif self.reduction == "sum":
            loss = weighted_errors.sum()
        else:
            loss = weighted_errors

        result = model_out.copy()
        result["loss"] = loss
        return result


class CombinedLoss(AbstractLoss):
    """
    Combine multiple loss functions into a weighted objective.

    Parameters
    ----------
    losses : dict[str, AbstractLoss]
        Mapping from loss names to loss instances.
    weights : dict[str, float] | None, default=None
        Optional dictionary mapping loss names to weights.
        If None, all losses are weighted equally.
    """

    def __init__(
        self,
        losses: Dict[str, AbstractLoss],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the composite loss wrapper.

        Parameters
        ----------
        losses : dict[str, AbstractLoss]
            Mapping from loss names to loss instances.
        weights : dict[str, float] | None, default=None
            Optional dictionary mapping loss names to weights.
            If None, all losses are weighted equally.
        """
        super().__init__()
        self.losses = losses
        self.weights = weights or {name: 1.0 for name in losses.keys()}

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {name: w / total_weight for name, w in self.weights.items()}

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the weighted sum of the configured losses.

        Parameters
        ----------
        model_out : dict[str, torch.Tensor]
            Model outputs containing the tensors required by each loss.
        batch : dict[str, torch.Tensor]
            Batch data containing the tensors required by each loss.

        Returns
        -------
        dict[str, torch.Tensor]
            Copy of ``model_out`` with the total ``loss`` and individual losses.
        """
        total_loss = 0.0
        individual_losses = {}

        for name, loss_fn in self.losses.items():
            loss_result = loss_fn.forward(model_out, batch)
            individual_loss = loss_result["loss"]
            weight = self.weights[name]

            total_loss += weight * individual_loss
            individual_losses[f"{name}_loss"] = individual_loss

        result = model_out.copy()
        result["loss"] = total_loss
        result.update(individual_losses)
        return result
