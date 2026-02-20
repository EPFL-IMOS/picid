import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    """
    A residual block for the encoder, composed of Conv1d, GroupNorm, Dropout,
    and a skip connection.

    This "clever" version allows for configurable normalization and activation
    layers and uses nn.Identity() for disabled components.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout_prob: float = 0.0,
        norm_type: str = "group",  # "group", "batch", or None
        activation: str = "relu",  # "relu", "gelu", or "silu"
    ):
        super(ResidualBlock, self).__init__()
        self.dilation = dilation
        self.stride = stride

        # 1. Convolution (padding is handled in forward)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

        # 2. "Clever" Normalization
        if norm_type == "group":
            # GroupNorm(1, ...) is like InstanceNorm, good for varying seq lengths
            self.norm = nn.GroupNorm(1, out_channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm_type is None or norm_type.lower() == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # 3. "Clever" Activation
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu" or activation == "swish":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 4. "Clever" Dropout
        # nn.Dropout(0) is an Identity, but this is more explicit
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        # 5. "Clever" Skip Connection (logic is already good)
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
        identity = x

        # --- Manual "same" padding logic (corrected for dilation) ---
        k = self.conv.kernel_size[0]
        d = self.dilation

        pad_total = d * (k - 1)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        padded_x = F.pad(x, (pad_left, pad_right))
        # --- End of padding logic ---

        # Apply the layers in order
        out = self.conv(padded_x)
        out = self.norm(out)
        out = self.act(out)  # <-- Use the configurable activation
        out = self.dropout(out)

        if self.skip_connection:
            identity = self.skip_connection(x)

        return out + identity


class ResidualTransposeBlock(nn.Module):
    """
    A symmetrical residual block for the decoder, using ConvTranspose1d.
    This "clever" version is robust to all kernel sizes and strides
    and supports configurable normalization and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout_prob: float,
        dilation: int,
        norm_type: str = "group",  # "group", "batch", or None
        activation: str = "relu",  # "relu", "gelu", or "silu"
    ):
        super(ResidualTransposeBlock, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2
        output_padding = stride - 1

        # --- 1. Build main layers manually ---
        self.conv_t = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
        )

        # 2. "Clever" Normalization
        if norm_type == "group":
            self.norm = nn.GroupNorm(1, out_channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm_type is None or norm_type.lower() == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # 3. "Clever" Activation
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu" or activation == "swish":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 4. "Clever" Dropout
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        # --- 5. Build skip connection ---
        self.skip_connection = (
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                output_padding=output_padding,
                dilation=dilation,
            )
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x):
        identity = x

        # Apply main layers
        out = self.conv_t(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        # This trims the output by one element only when needed.
        # (This logic is unchanged from your original)
        if self.kernel_size % 2 == 0:
            if self.stride > 1:
                out = out[..., :-1]
            else:
                target_len = x.shape[-1]
                out = out[..., :target_len]

        if self.skip_connection:
            identity = self.skip_connection(x)
            # Apply the same robust logic to the skip connection
            if self.kernel_size % 2 == 0:
                if self.stride > 1:
                    identity = identity[..., :-1]
                else:
                    target_len = x.shape[-1]
                    identity = identity[..., :target_len]

        return out + identity


class SymmetricalAutoencoder(nn.Module):
    """
    A symmetrical convolutional autoencoder built from residual blocks.
    The architecture is defined by a configuration dictionary.
    """

    def __init__(self, config):
        super(SymmetricalAutoencoder, self).__init__()
        self.config = config

        # --- Prepare Block Configurations ---
        input_channels = int(config["input_channels"])
        output_channels = [int(c) for c in config["output_channels"]]
        kernels = [int(k) for k in config["kernels"]]
        strides = [int(s) for s in config["strides"]]
        dilations = [int(d) for d in config["dilations"]]

        # --- Get global "clever" args from config ---
        # These will be passed to every block
        dropout_prob = float(config.get("dropout_prob", 0.0))
        norm_type = config.get("norm_type", "group")
        activation = config.get("activation", "relu")
        # ---

        assert (
            len(output_channels) == len(kernels) == len(strides) == len(dilations)
        ), "All configuration lists must have the same length."

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

        # --- Build Encoder (Updated) ---
        encoder_blocks = []
        for block_cfg in blocks_config:
            encoder_blocks.append(
                ResidualBlock(
                    # Per-block settings
                    in_channels=int(block_cfg["in"]),
                    out_channels=int(block_cfg["out"]),
                    kernel_size=int(block_cfg["kernel"]),
                    stride=int(block_cfg["stride"]),
                    dilation=int(
                        block_cfg["dilation"]
                    ),  # <-- FIX 1: Use per-block dilation
                    # Global settings
                    dropout_prob=dropout_prob,
                    norm_type=norm_type,
                    activation=activation,
                )
            )
        self.encoder_conv = nn.Sequential(*encoder_blocks)

        # Calculate flattened size dynamically
        self._calculate_flattened_size()

        # --- FC Layers (Unchanged) ---
        self.encoder_fc = nn.Linear(self.flattened_size, int(config["latent_dim"]))
        self.decoder_fc = nn.Linear(int(config["latent_dim"]), self.flattened_size)

        # --- Build Decoder (Updated) ---
        decoder_blocks = []
        for block_cfg in reversed(blocks_config):
            decoder_blocks.append(
                ResidualTransposeBlock(
                    # Per-block settings
                    in_channels=int(block_cfg["out"]),
                    out_channels=int(block_cfg["in"]),
                    kernel_size=int(block_cfg["kernel"]),
                    stride=int(block_cfg["stride"]),
                    dilation=int(
                        block_cfg["dilation"]
                    ),  # <-- FIX 1: Use per-block dilation
                    # Global settings
                    dropout_prob=dropout_prob,
                    norm_type=norm_type,
                    activation=activation,
                )
            )
        self.decoder_conv = nn.Sequential(*decoder_blocks)

    def _calculate_flattened_size(self):
        """Passes a dummy tensor through the conv encoder to determine the flattened size."""
        dummy_input = torch.randn(
            1, int(self.config["input_channels"]), int(self.config["input_seq_len"])
        )
        dummy_output = self.encoder_conv(dummy_input)
        self.flattened_size = int(torch.numel(dummy_output) / dummy_output.shape[0])
        self.unflatten_shape = dummy_output.shape[1:]

    # --- encode, decode, and forward methods are unchanged ---

    def encode(self, x):
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.encoder_conv(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        latent = self.encoder_fc(x)
        return latent

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.reshape(x.size(0), *self.unflatten_shape)  # Unflatten
        x = self.decoder_conv(x)
        # Permute back to (batch, length, channels)
        reconstructed = x.permute(0, 2, 1)
        return reconstructed

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)


# --- Example Usage ---
if __name__ == "__main__":
    config = {
        "input_channels": 3,
        "input_seq_len": 256,
        "latent_dim": 32,
        "dropout_prob": 0.1,
        "blocks_config": [
            # Encoder blocks: (in_channels, out_channels, kernel_size, stride)
            {"in": 3, "out": 16, "kernel": 7, "stride": 2},
            {"in": 16, "out": 32, "kernel": 5, "stride": 2},
            {"in": 32, "out": 64, "kernel": 3, "stride": 2},
        ],
    }

    model = SymmetricalAutoencoder(config).to(device)
    print("--- Symmetrical Autoencoder Architecture ---")
    print(model)
    print("-" * 60)

    # --- Test with a dummy tensor ---
    DUMMY_BATCH_SIZE = 32
    dummy_input = torch.randn(
        DUMMY_BATCH_SIZE, config["input_seq_len"], config["input_channels"]
    ).to(device)

    print(f"Input shape:  {dummy_input.shape}")
    reconstructed_output = model(dummy_input)
    print(f"Output shape: {reconstructed_output.shape}")

    assert dummy_input.shape == reconstructed_output.shape
    print("\nSuccessfully tested the autoencoder. Input and output shapes match.")

    encoded = model.encode(dummy_input)
    print(f"Encoded shape: {encoded.shape}")
