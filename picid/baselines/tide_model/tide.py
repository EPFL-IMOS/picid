import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer normalization with optional bias.

    PyTorch's built-in :class:`~torch.nn.LayerNorm` always includes a bias
    parameter. This small wrapper keeps the same runtime behavior while allowing
    bias-free checkpoints and configs to be represented explicitly.

    Parameters
    ----------
    ndim : int
        Number of normalized dimensions, i.e. the feature width.
    bias : bool
        If ``True``, allocate a learnable bias parameter.
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        Apply layer normalization.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape ``(..., ndim)``.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as ``input``.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-2)


class ResBlock(nn.Module):
    """
    Residual block with two linear layers and a skip connection.

    The block mirrors the TiDE residual MLP pattern: a first projection expands
    the input, a second projection contracts it to the output width, and a
    direct skip path preserves the original signal geometry.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    output_dim : int
        Output feature dimension.
    dropout : float, optional
        Dropout probability applied between the two projections.
    bias : bool, optional
        Whether to use bias in the linear layers.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)
        # self.ln = RMSNorm(output_dim, bias=bias)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, output_dim).
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


# TiDE
class Model(nn.Module):
    """
    TiDE: Time-series Dense Encoder.

    This module exposes the baseline TiDE architecture used for forecasting,
    imputation, regression, and classification tasks.

    Parameters
    ----------
    configs : object
        Configuration with attributes such as ``task_name``, ``seq_len``,
        ``pred_len``, ``freq``, ``feature_dim``, ``feature_encode_dim``,
        ``d_model``, ``e_layers``, ``d_layers``, ``d_ff``, ``dropout``,
        ``c_out``, and optionally ``num_class``.
    bias : bool, optional
        Whether to use bias in linear layers.
    """

    def __init__(self, configs, bias=True):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  # L
        # self.label_len = configs.label_len
        self.pred_len = configs.pred_len  # H
        self.hidden_dim = configs.d_model
        self.res_hidden = configs.d_model
        self.encoder_num = configs.e_layers
        self.decoder_num = configs.d_layers
        self.freq = configs.freq
        self.feature_encode_dim = configs.feature_encode_dim
        self.decode_dim = configs.c_out
        self.temporalDecoderHidden = configs.d_ff
        self.num_class = configs.get("num_class", None)

        dropout = configs.dropout

        assert (configs.freq is None) or (configs.feature_dim is None)

        if configs.feature_dim is not None:
            self.feature_dim = configs.feature_dim
        else:
            freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
            self.feature_dim = freq_map[self.freq]

        if self.task_name not in ["state_forecasting"]:
            flatten_dim = (
                self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim
            )
            temporal_decoder_in_dim = self.decode_dim + self.feature_encode_dim
        else:
            flatten_dim = self.seq_len
            temporal_decoder_in_dim = self.decode_dim

        self.feature_encoder = ResBlock(
            self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, bias
        )
        self.encoders = nn.Sequential(
            ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias),
            *(
                [
                    ResBlock(
                        self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias
                    )
                ]
                * (self.encoder_num - 1)
            ),
        )
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
            or self.task_name == "state_forecasting"
        ):
            self.decoders = nn.Sequential(
                *(
                    [
                        ResBlock(
                            self.hidden_dim,
                            self.res_hidden,
                            self.hidden_dim,
                            dropout,
                            bias,
                        )
                    ]
                    * (self.decoder_num - 1)
                ),
                ResBlock(
                    self.hidden_dim,
                    self.res_hidden,
                    self.decode_dim * self.pred_len,
                    dropout,
                    bias,
                ),
            )
            self.temporalDecoder = ResBlock(
                temporal_decoder_in_dim,
                self.temporalDecoderHidden,
                1,
                dropout,
                bias,
            )
            self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        if self.task_name == "imputation":
            self.decoders = nn.Sequential(
                *(
                    [
                        ResBlock(
                            self.hidden_dim,
                            self.res_hidden,
                            self.hidden_dim,
                            dropout,
                            bias,
                        )
                    ]
                    * (self.decoder_num - 1)
                ),
                ResBlock(
                    self.hidden_dim,
                    self.res_hidden,
                    self.decode_dim * self.seq_len,
                    dropout,
                    bias,
                ),
            )
            self.temporalDecoder = ResBlock(
                self.decode_dim + self.feature_encode_dim,
                self.temporalDecoderHidden,
                1,
                dropout,
                bias,
            )
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)

        if self.task_name == "classification":
            self.head = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.num_class),
            )

        if self.task_name == "regression":
            self.head = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, 1),
            )

    def forecast(self, x_enc, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        if batch_y_mark is not None:
            feature = self.feature_encoder(batch_y_mark)
            hidden = self.encoders(
                torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1)
            )
        else:
            hidden = self.encoders(x_enc)

        decoded = self.decoders(hidden).reshape(
            hidden.shape[0], self.pred_len, self.decode_dim
        )

        if batch_y_mark is not None:
            temp_dec_in = torch.cat([feature[:, self.seq_len :], decoded], dim=-1)
        else:
            temp_dec_in = decoded
        dec_out = self.temporalDecoder(temp_dec_in).squeeze(-1) + self.residual_proj(
            x_enc
        )
        # print(feature.sum().item(), hidden.sum().item(), decoded.sum().item(), dec_out.sum().item())

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        return dec_out

    def classification_and_regression(self, x_enc, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        feature = self.feature_encoder(batch_y_mark)
        hidden = self.encoders(
            torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1)
        )

        return hidden

    def forward(self, x_enc, x_mark_enc, x_dec, batch_y_mark, mask=None):
        """
        Forward pass.

        Parameters
        ----------
        x_enc : torch.Tensor
            Encoder input: shape (batch, seq_len, channels).
        x_mark_enc : torch.Tensor or None
            Exogenous dynamic features for encoder; shape (batch, seq_len, feature_dim)
            for forecast, or None for state_forecasting.
        x_dec : torch.Tensor or None
            Decoder input (unused in TiDE).
        batch_y_mark : torch.Tensor or None
            Exogenous features for decoder; shape (batch, seq_len+pred_len, feature_dim)
            for forecast, or None for state_forecasting.
        mask : torch.Tensor or None, optional
            Optional mask (unused).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, pred_len, c_out) for forecast tasks,
            or (batch, 1, num_class) for classification.
        """
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            batch_y_mark = torch.concat(
                [x_mark_enc, batch_y_mark[:, -self.pred_len :, :]], dim=1
            )
            dec_out = torch.stack(
                [
                    self.forecast(x_enc[:, :, feature], batch_y_mark)
                    for feature in range(x_enc.shape[-1])
                ],
                dim=-1,
            )
            return dec_out  # [B, L, D]

        if self.task_name == "state_forecasting":
            dec_out = torch.stack(
                [
                    self.forecast(x_enc[:, :, feature], None)
                    for feature in range(x_enc.shape[-1])
                ],
                dim=-1,
            )
            return dec_out  # [B, L, D]

        if self.task_name == "classification" or self.task_name == "regression":
            batch_y_mark = torch.concat(
                [x_mark_enc, batch_y_mark[:, -self.pred_len :, :]], dim=1
            )
            dec_out = torch.stack(
                [
                    self.classification_and_regression(
                        x_enc[:, :, feature], batch_y_mark
                    )
                    for feature in range(x_enc.shape[-1])
                ],
                dim=-1,
            )

            if self.task_name == "regression":
                dec_out = self.head(dec_out)  # [B, D]
            else:
                dec_out = self.head(dec_out)

            return dec_out.unsqueeze(1)  # [B, L, D]
