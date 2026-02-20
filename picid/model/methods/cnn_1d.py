import torch
import torch.nn as nn
from lmetk.model.autoencoder.residual_1d import ResidualBlock


class EncoderModel(nn.Module):
    """
    (Docstring unchanged...)
    """

    def __init__(self, config, task_type: str, num_classes: int = 1):
        super().__init__()
        self.config = config
        self.task_type = task_type
        self.latent_dim = int(config["latent_dim"])

        # --- 1. Build Encoder Backbone ---
        # (This logic is unchanged)
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

        # --- 2. Build Latent Head (WITH NORMALIZATION) ---
        self._calculate_flattened_size()

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.flattened_size, self.latent_dim),
            # --- THIS IS THE FIX ---
            # Normalizes the features *before* the activation
            nn.LayerNorm(self.latent_dim),
            # ---------------------
            torch.nn.ReLU(),
        )

        # --- 3. Build FINAL Output Head ---
        # (This logic is unchanged)
        if self.task_type == "regression":
            self.final_head = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, 1), torch.nn.Flatten(start_dim=0)
            )
        elif self.task_type == "classification":
            if num_classes <= 0:
                raise ValueError("num_classes must be positive for classification.")
            self.final_head = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, num_classes)
            )
        else:
            raise ValueError(
                f"Unknown task_type: '{self.task_type}'. Must be 'regression' or 'classification'."
            )

        # --- 4. APPLY WEIGHT INITIALIZATION ---
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Applies Kaiming and Zero-initialization to the model.
        """
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # --- ADDED: Init LayerNorm as well ---
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, ResidualBlock):
            if m.skip_connection is None:
                if hasattr(m, "norm") and hasattr(m.norm, "weight"):
                    nn.init.constant_(m.norm.weight, 0)

    def _calculate_flattened_size(self):
        # (Unchanged)
        dummy_input = torch.randn(
            1, int(self.config["input_channels"]), int(self.config["input_seq_len"])
        )
        dummy_output = self.cnn_backbone(dummy_input)
        self.flattened_size = int(torch.numel(dummy_output) / dummy_output.shape[0])

    def forward(self, x):
        # (Unchanged)
        feat = self.cnn_backbone(x)
        feat = self.head(feat)
        output = self.final_head(feat)
        return output
