import torch
from torch import nn

from picid.model.forecasters.shared_layers.Embed_PatchTST import (
    PatchEmbedding,
)
from picid.model.forecasters.shared_layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
)
from picid.model.forecasters.shared_layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)


class Transpose(nn.Module):
    """
    Transpose tensor dimensions.

    Parameters
    ----------
    *dims : int
        Dimensions to transpose.
    contiguous : bool, optional
        If True, call .contiguous() after transpose. Default is False.
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        """
        Transpose an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transposed tensor.
        """
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    """
    Flatten encoder output and project it to the target window.

    Parameters
    ----------
    nf : int
        Input feature size (d_model * patch_num after flatten).
    target_window : int
        Output sequence length.
    head_dropout : float, optional
        Dropout probability. Default is 0.
    """

    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Project a flattened encoder output to the target window.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[bs, nvars, d_model, patch_num]``.

        Returns
        -------
        torch.Tensor
            Output of shape ``[bs, nvars, target_window]``.
        """
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    PatchTST baseline model.

    Paper link: https://arxiv.org/pdf/2211.14730.pdf

    Parameters
    ----------
    configs : object
        Configuration with task_name, seq_len, pred_len, enc_in, d_model,
        dropout, e_layers, d_ff, n_heads, activation, factor, num_class
        (classification), num_reg_targets (regression).
    patch_len : int, optional
        Patch length for patch_embedding. Default is 16.
    stride : int, optional
        Stride for patch_embedding. Default is 8.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)  # noqa: E741
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)
            ),
        )

        # Prediction Head
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.head = FlattenHead(
                self.head_nf,
                configs.pred_len,
                head_dropout=configs.dropout,
            )
        elif self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.head = FlattenHead(
                self.head_nf,
                configs.seq_len,
                head_dropout=configs.dropout,
            )
        elif self.task_name == "classification":
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class
            )
        elif self.task_name == "regression":
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_reg_targets
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def regression(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_reg_targets)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Dispatch to the task-specific forward branch.

        Parameters
        ----------
        x_enc : torch.Tensor
            Encoder input of shape [B, L, D].
        x_mark_enc : torch.Tensor, optional
            Encoder time features (unused).
        x_dec : torch.Tensor, optional
            Decoder input (unused).
        x_mark_dec : torch.Tensor, optional
            Decoder time features (unused).
        mask : torch.Tensor, optional
            Attention mask. Default is None.

        Returns
        -------
        torch.Tensor or None
            For forecast: ``[B, pred_len, D]``.
            For regression: ``[B, num_reg_targets]``.
            For classification: ``[B, num_class]``.
        """
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        elif self.task_name == "regression":
            dec_out = self.regression(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        elif self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
