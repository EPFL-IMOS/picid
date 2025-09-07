"""Abstract class that provides an interface to handle the network output."""

from abc import ABC, abstractmethod

from einops import rearrange
import numpy as np
import torch
from lightning_fabric.utilities.data import AttributeDict
from tqdm import tqdm


class AbstractFitPredictWrapper(ABC):
    r"""Abstract class that provides an interface to handle the network output.

    Parameters
    ----------
    backbone : Sklearn
        Backbone model
    """

    def __init__(
        self,
        backbone,
        yield_strategy: bool = False,
        yield_batch_size: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.kwargs = AttributeDict(kwargs)
        self.yield_strategy = yield_strategy
        self.yield_batch_size = yield_batch_size

        # Check if backbone supports fit predict
        if not hasattr(self.backbone, "fit") or not hasattr(self.backbone, "predict"):
            raise ValueError("Backbone model must implement fit and predict methods.")

    def fit(self, X, y):
        self.backbone.fit(X.cpu(), y.cpu())

    def predict(self, X):

        X = X.cpu()
        if self.yield_strategy:
            batch_size = self.yield_batch_size
            n_samples = X.shape[0]
            outputs_list = []
            for start in tqdm(
                range(0, n_samples, batch_size),
                desc=f"Predicting in batches of size {batch_size}",
            ):
                end = min(start + batch_size, n_samples)
                batch_X = X[start:end]
                batch_outputs = self.backbone.predict(batch_X)
                outputs_list.append(batch_outputs)
            outputs = np.concatenate(outputs_list, axis=0)
            y = torch.from_numpy(outputs)
        else:
            y = self.backbone.predict(X)

        if not isinstance(y, torch.Tensor):
            if hasattr(y, "__array__"):
                y = torch.from_numpy(y)
            else:
                raise TypeError(
                    "Predicted output must be a torch.Tensor or numpy.ndarray."
                )

        if y.ndim == 1:
            y = rearrange(y, "a -> a 1")
        assert y.ndim == 2, "Predicted output must be 2-dimensional."
        return y

    @abstractmethod
    def serialize_model(self, task_id):
        """Serialize the model to a file."""
        pass

    @abstractmethod
    def load_model(self, task_id):
        """Load the model from a file."""
        pass

    @property
    @abstractmethod
    def allows_multi_target(self) -> bool:
        """Indicates whether the model supports multi-target prediction."""
        pass


class AbstractFeedForwardWrapper(ABC, torch.nn.Module):
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


class AbstractFeedForwardTrainingWrapper(ABC, torch.nn.Module):
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
