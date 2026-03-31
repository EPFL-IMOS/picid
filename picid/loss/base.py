"""
Abstract base class for loss functions in PyTorch.

This module provides an abstract base class for implementing custom loss functions
for transformer models using standard PyTorch tensors and data structures.
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch


class AbstractLoss(ABC):
    """
    Abstract base class for loss functions.

    This class defines the standard interface for implementing custom loss functions
    for transformer models. Subclasses must implement the forward method to provide
    specific loss computation logic.
    """

    def __init__(self) -> None:
        """Initialize the abstract loss function."""
        super().__init__()

    def __call__(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for a single model batch.

        This wrapper keeps the public loss interface uniform and delegates the
        actual implementation to :meth:`forward` so subclasses only need to
        define the task-specific loss logic.

        Parameters
        ----------
        model_out : Dict[str, torch.Tensor]
            Dictionary containing the model output tensors.
        batch : Dict[str, torch.Tensor]
            Dictionary containing the batch data tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the original model output with the computed loss.
        """
        return self.forward(model_out, batch)

    @abstractmethod
    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for loss computation.

        This method must be implemented by subclasses to define the specific
        loss computation logic.

        Parameters
        ----------
        model_out :
            Dictionary containing the model output tensors.
            Typically includes keys like 'predictions', 'logits', etc.
        batch :
            Dictionary containing the batch data tensors.
            Typically includes keys like 'targets', 'labels', 'input_ids', etc.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the original model output with added loss information.
            Should include at least a 'loss' key with the computed loss tensor.
        """
        pass
