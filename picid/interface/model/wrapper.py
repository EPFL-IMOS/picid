from typing import Callable

from torch import nn


def empty_function(r):
    return r

class ModelWrapper(nn.Module):
    """

    A transparent wrapper for PyTorch models that integrates optional
    pre-processing and post-processing steps into the forward pass.

    This class acts as a proxy for the underlying base model. Any attributes
    or methods not explicitly defined in this wrapper will be delegated to
    the base model, allowing it to be used as a drop-in replacement.

    Attributes:
        base_model (nn.Module): The underlying PyTorch model being wrapped.
        _post_process_function (Callable): The function applied to the model's output.
        _pre_process_function (Callable): The function applied to the input before
            passing it to the model.
    """

    def __init__(self,
                 model : nn.Module,
                 post_process_function: Callable | None = None,
                 pre_process_function:  Callable | None = None,
                 **kwargs):
        """Initialise the wrapper. See class docstring for parameter details."""

        super().__init__(**kwargs)
        self.base_model = model

        if post_process_function is None:
            post_process_function = empty_function

        self._post_process_function = post_process_function

        if pre_process_function is None:
            pre_process_function = empty_function

        self._pre_process_function = pre_process_function

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.base_model, item)

    def forward(self, x, **kwargs):
        """Run pre-processing, base model, and post-processing in sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor from the dataloader batch.
        **kwargs
            Additional keyword arguments (not forwarded to the base model).

        Returns
        -------
        torch.Tensor
            Output of the base model after post-processing.
        """
        x = self._pre_process_function(x)
        pred = self.base_model(x)
        pred = self._post_process_function(pred)

        return pred
