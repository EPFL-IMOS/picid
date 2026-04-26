import torch
import torch.nn as nn
from typing import Dict, Any, List


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) baseline with configurable depth and compression.

    This model flattens the input sequence and passes it through a series of
    linear layers. Each hidden layer is followed by Layer Normalization and ReLU.

    Architecture (num_layers=L):
        Flatten -> [Linear(In->Hidden) -> LN -> ReLU] (Layer 1)
                -> [Linear(Hidden->Hidden) -> LN -> ReLU] (Layers 2..L-1)
                -> Linear(Hidden->Out) (Layer L)

    Attributes
    ----------
    input_dim : int
        Seq_Len * Input_Channels.
    hidden_dim : int
        Size of the hidden layers.
    output_dim : int
        Number of regression targets or classification classes.
    num_layers : int
        Total number of linear transformations.
    """

    def __init__(self, config: Dict[str, Any], task_type: str, num_targets: int):
        """
        Create the MLP backbone from the provided configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Input-shape configuration.
        task_type : str
            Project task identifier.
        num_targets : int
            Number of regression targets or classification classes.
        """
        super().__init__()

        # 1. Dimensions
        self.input_dim = int(config["seq_len"]) * int(config["input_channels"])
        self.hidden_dim = int(config.get("hidden_dim", 64))
        self.output_dim = num_targets

        # Default to 2 layers (1 hidden layer) if not specified
        self.num_layers = int(config.get("num_layers", 2))

        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")

        # 2. Build Layers
        layers: List[nn.Module] = []

        if self.num_layers == 1:
            # Simple Linear Map: In -> Out
            layers.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            # --- First Hidden Layer (In -> Hidden) ---
            layers.extend(
                [
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                ]
            )

            # --- Intermediate Hidden Layers (Hidden -> Hidden) ---
            # We add (num_layers - 2) intermediate blocks
            for _ in range(self.num_layers - 2):
                layers.extend(
                    [
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                        nn.ReLU(),
                    ]
                )

            # --- Final Layer (Hidden -> Out) ---
            layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Kaiming normal weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x_flat = x.reshape(b, -1)
        return self.net(x_flat)
