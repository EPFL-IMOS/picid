"""1D residual building blocks for the CNN estimator."""

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Build a 1D residual block with optional normalization and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    stride : int, default=1
        Convolution stride.
    dilation : int, default=1
        Convolution dilation.
    dropout_prob : float, default=0.0
        Dropout probability applied after the activation.
    norm_type : str, default="group"
        Normalization type: ``"group"``, ``"batch"``, or ``None``/``"none"``.
    activation : str, default="relu"
        Activation type: ``"relu"``, ``"gelu"``, ``"silu"``, or ``"swish"``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout_prob: float = 0.0,
        norm_type: str = "group",
        activation: str = "relu",
    ):
        super(ResidualBlock, self).__init__()
        self.dilation = dilation
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

        if norm_type == "group":
            self.norm = nn.GroupNorm(1, out_channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm_type is None or norm_type.lower() == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu" or activation == "swish":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.skip_connection = (
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x):
        """
        Apply residual convolution block with optional skip connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Output tensor of the same batch size and updated channel count.
        """
        identity = x

        k = self.conv.kernel_size[0]
        d = self.dilation
        pad_total = d * (k - 1)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        padded_x = F.pad(x, (pad_left, pad_right))

        out = self.conv(padded_x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        if self.skip_connection:
            identity = self.skip_connection(x)

        return out + identity
