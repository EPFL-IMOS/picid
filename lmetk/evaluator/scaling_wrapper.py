"""Utilities for handling data scaling and inverse scaling."""

from einops import rearrange
import torch
import numpy as np
from typing import Optional, Any, Tuple

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.multisource import InverseTransformMixin


class ScalingWrapper:
    """Wrapper class to handle inverse scaling operations."""

    def __init__(
        self,
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        task_mode: bool = False,
    ):
        """
        Initialize the scaling wrapper.

        Parameters
        ----------
        inverse_transform : Optional[Any]
            The scaler object with inverse_transform method
        apply_inverse : bool
            Whether to apply inverse scaling
        """
        self.inverse_transform = inverse_transform
        self.apply_inverse = apply_inverse_scaling
        self.task_mode = task_mode

    def inverse_transform_if_needed(
        self,
        predictions: Tuple[torch.Tensor, np.ndarray],
        targets: Tuple[torch.Tensor, np.ndarray],
    ) -> tuple:
        """
        Apply inverse transform to predictions and targets if needed.

        Parameters
        ----------
        predictions : np.ndarray or torch.Tensor
            Model predictions
        targets : np.ndarray or torch.Tensor
            Ground truth targets

        Returns
        -------
        tuple
            (transformed_predictions, transformed_targets)
        """
        if not (self.inverse_transform and self.apply_inverse):
            return predictions, targets

        # Check if the incomming data is torch tensor, is so, then covert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().detach().numpy()

        # Apply inverse transform while preserving original shapes
        pred_shape = predictions.shape
        target_shape = targets.shape

        if len(pred_shape) == 3:

            assert (
                pred_shape[0] == 1 or self.task_mode
            ), f"Expected batch size of 1 for 3D predictions, got {pred_shape[0]}"

            # Apply inverse transform (assuming scaler expects 2D input)
            predictions_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(predictions=predictions.squeeze(0))),
                metadata={},
            ).reshape(pred_shape)

            targets_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(targets=targets.squeeze(0))), metadata={}
            ).reshape(target_shape)

        elif len(pred_shape) == 2:
            # Apply inverse transform (assuming scaler expects 2D input)
            predictions_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(predictions=predictions)), metadata={}
            ).reshape((1, -1))

            targets_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(targets=targets)), metadata={}
            ).reshape((1, -1))
        else:
            raise ValueError(f"Unexpected shape: {pred_shape}")

        return predictions_transformed, targets_transformed


class MultivariateTimeseriesScalingWrapper:
    """Wrapper class to handle inverse scaling operations."""

    def __init__(
        self,
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        task_mode: bool = False,
    ):
        """
        Initialize the scaling wrapper.

        Parameters
        ----------
        inverse_transform : Optional[Any]
            The scaler object with inverse_transform method
        apply_inverse : bool
            Whether to apply inverse scaling
        """
        self.inverse_transform = inverse_transform
        self.apply_inverse = apply_inverse_scaling
        self.task_mode = task_mode

        assert (
            task_mode == "multivariate"
        ), "MultivariateTimeseriesScalingWrapper only supports multivariate task_mode"

    def inverse_transform_if_needed(
        self,
        predictions: Tuple[torch.Tensor, np.ndarray],
        targets: Tuple[torch.Tensor, np.ndarray],
    ) -> tuple:
        """
        Apply inverse transform to predictions and targets if needed.

        Parameters
        ----------
        predictions : np.ndarray or torch.Tensor
            Model predictions
        targets : np.ndarray or torch.Tensor
            Ground truth targets

        Returns
        -------
        tuple
            (transformed_predictions, transformed_targets)
        """
        if not (self.inverse_transform and self.apply_inverse):
            return predictions, targets

        # Check if the incomming data is torch tensor, is so, then covert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().detach().numpy()

        # Apply inverse transform while preserving original shapes
        pred_shape = predictions.shape

        if len(pred_shape) == 3:

            assert (
                pred_shape[0] == 1 or self.task_mode
            ), f"Expected batch size of 1 for 3D predictions, got {pred_shape[0]}"

            preds_flattened = rearrange(predictions, "b t c -> (b t) c")
            targets_flattened = rearrange(targets, "b t c -> (b t) c")

            # Apply inverse transform (assuming scaler expects 2D input)
            predictions_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(predictions=preds_flattened)),
                metadata={},
            )

            targets_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(targets=targets_flattened)), metadata={}
            )

            targets_unflattened = rearrange(
                targets_transformed, "(b t) c -> b t c", b=pred_shape[0]
            )

            predictions_unflattened = rearrange(
                predictions_transformed, "(b t) c -> b t c", b=pred_shape[0]
            )

        else:
            raise ValueError(f"Unexpected shape: {pred_shape}, need to be 3D")

        return predictions_unflattened, targets_unflattened
