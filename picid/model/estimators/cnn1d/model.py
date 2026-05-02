"""CNN 1D encoder model with task-specific output head."""

import torch
import torch.nn as nn

from picid.model.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from picid.model.estimators.cnn1d.blocks import ResidualBlock


class EncoderModel(nn.Module):
    """
    Build the convolutional encoder and task-specific output head.

    Parameters
    ----------
    config : dict
        Configuration dictionary defining the encoder blocks.
    task_type : str
        Project task identifier.
    num_classes : int, default=1
        Number of output classes used for classification tasks.
    """

    def __init__(self, config, task_type: str, num_classes: int = 1):
        super().__init__()
        self.config = config
        self.task_type = task_type
        self.latent_dim = int(config["latent_dim"])

        input_channels = int(config["input_channels"])
        output_channels = [int(c) for c in config["output_channels"]]
        kernels = [int(k) for k in config["kernels"]]
        strides = [int(s) for s in config["strides"]]
        dilations = [int(d) for d in config["dilations"]]

        assert (
            len(output_channels) == len(kernels) == len(strides) == len(dilations)
        ), "Config lists must all have the same length."

        blocks_config = [
            {"in": in_ch, "out": out_ch, "kernel": k, "stride": s, "dilation": d}
            for in_ch, out_ch, k, s, d in zip(
                [input_channels] + output_channels[:-1],
                output_channels,
                kernels,
                strides,
                dilations,
            )
        ]

        dropout_prob = float(config.get("dropout_prob", 0.0))
        norm_type = config.get("norm_type", "group")
        activation = config.get("activation", "relu")

        encoder_blocks = []
        for block_cfg in blocks_config:
            encoder_blocks.append(
                ResidualBlock(
                    in_channels=int(block_cfg["in"]),
                    out_channels=int(block_cfg["out"]),
                    kernel_size=int(block_cfg["kernel"]),
                    stride=int(block_cfg["stride"]),
                    dilation=int(block_cfg["dilation"]),
                    dropout_prob=dropout_prob,
                    norm_type=norm_type,
                    activation=activation,
                )
            )
        self.cnn_backbone = nn.Sequential(*encoder_blocks)

        self._calculate_flattened_size()

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.flattened_size, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            torch.nn.ReLU(),
        )

        if self.task_type in REGRESSION_TASKS:
            self.final_head = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, 1), torch.nn.Flatten(start_dim=0)
            )
        elif self.task_type in CLASSIFICATION_TASKS:
            if num_classes <= 0:
                raise ValueError("num_classes must be positive for classification.")
            self.final_head = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, num_classes)
            )
        else:
            raise ValueError(
                f"Unknown task_type: '{self.task_type}'. Must be 'regression' or 'classification'."
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize convolutional and dense layers.

        Parameters
        ----------
        m : torch.nn.Module
            Module instance inspected during recursive initialization.
        """
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, ResidualBlock):
            if m.skip_connection is None:
                if hasattr(m, "norm") and hasattr(m.norm, "weight"):
                    nn.init.constant_(m.norm.weight, 0)

    def _calculate_flattened_size(self):
        """Infer flattened feature size by passing a dummy tensor through the backbone."""
        dummy_input = torch.randn(
            1, int(self.config["input_channels"]), int(self.config["input_seq_len"])
        )
        dummy_output = self.cnn_backbone(dummy_input)
        self.flattened_size = int(torch.numel(dummy_output) / dummy_output.shape[0])

    def forward(self, x):
        """
        Run encoder backbone and output head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Task prediction tensor.
        """
        feat = self.cnn_backbone(x)
        feat = self.head(feat)
        output = self.final_head(feat)
        return output
