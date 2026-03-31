import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class LinearBaseline(nn.Module):
    """
    A simple but robust linear baseline for PHM tasks.

    This model flattens the input time-series sequence and applies a single linear
    transformation. It serves as a lower-bound benchmark to determine if deep
    temporal architectures (RNNs, CNNs) are actually necessary.

    Equation:
        y = Wx + b

    Attributes
    ----------
    input_dim : int
        The total number of flattened input features.
    output_dim : int
        The dimension of the prediction (num_targets or num_classes).
    linear : nn.Linear
        The linear transformation layer.
    """

    def __init__(self, config: Dict[str, int], task_type: str, num_targets: int = 1):
        """
        Initializes the LinearBaseline.

        Parameters
        ----------
        config : Dict[str, int]
            Configuration dictionary containing:
            - 'seq_len': Length of the input time series.
            - 'input_channels': Number of sensor channels.
        task_type : str
            The type of task ('regression' or 'classification').
        num_targets : int, optional
            The output dimension.
            - For Univariate Regression: 1
            - For Multivariate Regression: N (number of targets)
            - For Classification: N (number of classes)
            Defaults to 1.
        """
        super(LinearBaseline, self).__init__()
        self.config = config
        self.task_type = task_type

        # Flatten inputs: (Seq_Len * Channels)
        self.input_dim = int(config["seq_len"]) * int(config["input_channels"])
        self.output_dim = num_targets

        # The core model
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        # Apply Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize linear layers with Kaiming normal weights.

        Parameters
        ----------
        m : nn.Module
            Module instance inspected during recursive initialization.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flatten the input sequence and apply the linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, seq_len)``.

        Returns
        -------
        torch.Tensor
            Prediction tensor.
        """
        # x: (Batch, Channels, Seq_Len) -> Flatten to (Batch, Features)
        b = x.size(0)
        x_flat = x.reshape(b, -1)

        return self.linear(x_flat)


class PolynomialBaseline(nn.Module):
    """
    A Polynomial regression baseline capable of modeling non-linear acceleration.

    This model explicitly expands the feature space with polynomial powers before
    applying a linear transformation. It is effective for degradation trends that
    accelerate over time (e.g., quadratic bearing wear).

    Equation:
        y = W * [x, x^2, ..., x^d] + b

    Attributes
    ----------
    degree : int
        The polynomial degree (d).
    expanded_dim : int
        The dimension of the expanded feature space.
    linear : nn.Linear
        The linear transformation layer.
    """

    def __init__(self, config: Dict[str, Any], task_type: str, num_targets: int = 1):
        """
        Initializes the PolynomialBaseline.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary. Must contain:
            - 'seq_len': Length of the input time series.
            - 'input_channels': Number of sensor channels.
            - 'poly_degree' (optional): The degree of the polynomial expansion (default: 2).
        task_type : str
            The type of task ('regression' or 'classification').
        num_targets : int, optional
            The output dimension (targets or classes).
        """
        super(PolynomialBaseline, self).__init__()
        self.config = config
        self.task_type = task_type
        self.output_dim = num_targets

        self.degree = int(config.get("poly_degree", 2))
        base_input_dim = int(config["seq_len"]) * int(config["input_channels"])

        # Feature Expansion: base_dim * degree
        # We concatenate powers side-by-side
        self.expanded_dim = base_input_dim * self.degree

        self.linear = nn.Linear(self.expanded_dim, self.output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize linear layers with Kaiming normal weights.

        Parameters
        ----------
        m : nn.Module
            Module instance inspected during recursive initialization.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand flattened features with powers and apply the linear head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, seq_len)``.

        Returns
        -------
        torch.Tensor
            Prediction tensor.
        """
        b = x.size(0)
        x_flat = x.reshape(b, -1)

        # Polynomial Expansion: [x^1, x^2, ..., x^degree]
        # We compute powers independently to keep memory usage tractable compared to full interaction terms.
        features = [torch.pow(x_flat, d) for d in range(1, self.degree + 1)]
        x_poly = torch.cat(features, dim=1)

        return self.linear(x_poly)


class ExponentialBaseline(nn.Module):
    """
    Fit an exponential regression baseline on flattened features.

    Parameters
    ----------
    config : Dict[str, int]
        Input-shape configuration.
    task_type : str
        Project task identifier.
    num_targets : int, default=1
        Number of output targets.
    """

    def __init__(self, config: Dict[str, int], task_type: str, num_targets: int = 1):
        """
        Initializes the ExponentialBaseline.

        Parameters
        ----------
        config : Dict[str, int]
            Configuration dictionary.
        task_type : str
            Must be 'regression'. Classification is not supported.
        num_targets : int, optional
            Number of regression targets. Defaults to 1.

        Raises
        ------
        ValueError
            If task_type is 'classification'.
        """
        super(ExponentialBaseline, self).__init__()
        self.config = config

        if task_type == "classification":
            raise ValueError(
                "ExponentialBaseline is designed for physical growth laws (regression) "
                "and does not support Classification."
            )

        self.task_type = task_type
        self.output_dim = num_targets
        self.input_dim = int(config["seq_len"]) * int(config["input_channels"])

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize linear layers with Xavier normal weights.

        Parameters
        ----------
        m : nn.Module
            Module instance inspected during recursive initialization.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear head and map outputs through ``exp``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, seq_len)``.

        Returns
        -------
        torch.Tensor
            Prediction tensor.
        """
        b = x.size(0)
        x_flat = x.reshape(b, -1)

        # Linear projection followed by exponential activation
        return torch.exp(self.linear(x_flat))
