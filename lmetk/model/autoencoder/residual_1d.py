import torch
import torch.nn as nn

# To load from a YAML file, you would use:
# import yaml
# with open("config.yaml", 'r') as f:
#     config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A residual block for the encoder, composed of Conv1d, GroupNorm, Dropout, and a skip connection.
    This version uses manual padding to support all kernel sizes and strides.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, dropout_prob, dilation
    ):
        super(ResidualBlock, self).__init__()

        # We will apply padding manually, so set padding=0 in the conv layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # <-- Set to 0
            dilation=dilation,
        )

        # Separate the rest of the layers from the convolution
        self.norm = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)

        # Skip connection to handle changes in dimensions
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

        # --- Manual "same" padding logic ---
        # 1. Get the kernel size from the conv layer
        k = self.conv.kernel_size[0]

        # 2. Calculate left and right padding
        # This handles both odd and even kernel sizes correctly
        pad_left = (k - 1) // 2
        pad_right = k - 1 - pad_left

        # 3. Apply padding to the input tensor
        padded_x = F.pad(x, (pad_left, pad_right))
        # --- End of padding logic ---

        # Pass the padded tensor through the layers
        out = self.conv(padded_x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.skip_connection:
            identity = self.skip_connection(x)

        # The output of the convolution and the skip connection will have the same
        # length because both are affected by the stride.
        return out + identity


class ResidualTransposeBlock(nn.Module):
    """
    A symmetrical residual block for the decoder, using ConvTranspose1d.
    This version is robust to all kernel sizes and strides.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, dropout_prob, dilation
    ):
        super(ResidualTransposeBlock, self).__init__()
        # Store parameters to use in the forward pass
        self.stride = stride
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2
        output_padding = stride - 1

        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
            ),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )

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
        out = self.layers(x)

        # This trims the output by one element only when needed.
        if self.kernel_size % 2 == 0:
            # For stride > 1, the target is upsampled. This trims the extra element.
            if self.stride > 1:
                out = out[..., :-1]
            # For stride = 1, the target is same-size. This also trims the extra element.
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

        # Check that all configuration lists have the same length
        assert len(output_channels) == len(kernels) == len(strides) == len(dilations), (
            f"All configuration lists (output_channels, kernels, strides, dilations) must have the same length. "
            f"Current lengths: {len(output_channels)}, {len(kernels)}, {len(strides)}, {len(dilations)}"
        )

        blocks_config = [
            {"in": in_ch, "out": out_ch, "kernel": k, "stride": s, "dilation": d}
            for in_ch, out_ch, k, s, d in zip(
                [input_channels]
                + output_channels[
                    :-1
                ],  # from input_channels: 3; and output_channels: [16, 32, 64]; It   Generates [3, 16, 32]
                output_channels,  # The list [16, 32, 64]
                kernels,
                strides,
                dilations,
            )
        ]
        encoder_blocks = []
        for block_cfg in blocks_config:
            encoder_blocks.append(
                ResidualBlock(
                    int(block_cfg["in"]),
                    int(block_cfg["out"]),
                    int(block_cfg["kernel"]),
                    int(block_cfg["stride"]),
                    float(config["dropout_prob"]),
                    int(config.get("dilation", 1)),
                )
            )
        self.encoder_conv = nn.Sequential(*encoder_blocks)

        # Calculate flattened size dynamically
        self._calculate_flattened_size()

        self.encoder_fc = nn.Linear(self.flattened_size, int(config["latent_dim"]))

        # --- Build Decoder ---
        self.decoder_fc = nn.Linear(int(config["latent_dim"]), self.flattened_size)

        decoder_blocks = []
        # Create decoder blocks in reverse order of encoder
        for block_cfg in reversed(blocks_config):
            decoder_blocks.append(
                ResidualTransposeBlock(
                    int(block_cfg["out"]),
                    int(block_cfg["in"]),
                    int(block_cfg["kernel"]),
                    int(block_cfg["stride"]),
                    float(config["dropout_prob"]),
                    int(config.get("dilation", 1)),
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
