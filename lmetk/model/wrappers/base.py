"""Abstract class that provides an interface to handle the network output."""

from abc import ABC, abstractmethod

import torch
from lightning_fabric.utilities.data import AttributeDict


class FitPredictWrapper(ABC):
    r"""Abstract class that provides an interface to handle the network output.

    Parameters
    ----------
    backbone : Sklearn
        Backbone model
    """

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.kwargs = AttributeDict(kwargs)

        # Check if backbone supports fit predict
        if not hasattr(self.backbone, "fit") or not hasattr(self.backbone, "predict"):
            raise ValueError("Backbone model must implement fit and predict methods.")

    def fit(self, X, y):
        self.backbone.fit(X.cpu(), y.cpu())

    def predict(self, X):
        return self.backbone.predict(X.cpu())


class AbstractWrapper(ABC, torch.nn.Module):
    r"""Abstract class that provides an interface to handle the network output.

    Parameters
    ----------
    backbone : torch.nn.Module
        Backbone model.
    **kwargs : dict
        Additional arguments for the class. It should contain the following keys:
        - out_channels (int): Number of output channels.
        - num_cell_dimensions (int): Number of cell dimensions.
    """

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.kwargs = AttributeDict(kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(backbone={self.backbone}, out_channels={self.backbone.out_channels}, dimensions={self.dimensions}, residual_connections={self.residual_connections})"

    def __call__(self, batch):
        r"""Forward pass for the model.

        This method calls the forward method and adds the residual connection.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        model_out = self.forward(batch)

        return model_out

    @abstractmethod
    def forward(self, batch):
        r"""Forward pass for the model.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        """
