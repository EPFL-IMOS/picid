import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding (fixed, non-trainable).

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int, optional
        Maximum sequence length. Defaults to 5000.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Return positional embeddings for the sequence length of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor; x.size(1) is the sequence length.

        Returns
        -------
        torch.Tensor
            Positional embeddings of shape (1, seq_len, d_model).
        """
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    Combine value, positional, and temporal embeddings for time series.

    Parameters
    ----------
    c_in : int
        Number of input channels/variates.
    d_model : int
        Embedding dimension.
    embed_type : str, optional
        Temporal embedding type: "fixed" or "timeF". Defaults to "fixed".
    freq : str, optional
        Time frequency: "h", "t", "s", "m", "a", "w", "d", "b". Defaults to "h".
    dropout : float, optional
        Dropout probability. Defaults to 0.1.
    """

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, L, c_in).
        x_mark : torch.Tensor
            Temporal features of shape (B, L, n_temporal_features).

        Returns
        -------
        torch.Tensor
            Embedded output of shape (B, L, d_model).
        """
        x = (
            self.value_embedding(x)
            + self.temporal_embedding(x_mark)
            + self.position_embedding(x)
        )
        return self.dropout(x)


class ContextEmbedding(nn.Module):
    """
    Embed value, context, positional, and temporal features.

    Parameters
    ----------
    c_in : int
        Number of input channels/variates.
    c_context : int
        Number of context channels.
    d_model : int
        Embedding dimension.
    embed_type : str, optional
        Temporal embedding type: "fixed" or "timeF". Defaults to "timeF".
    freq : str, optional
        Time frequency. Defaults to "h".
    dropout : float, optional
        Dropout probability. Defaults to 0.1.
    d_x_embedding_router : torch.Tensor, optional
        Mask to separate time features from context features in x_mark.
    """

    def __init__(
        self,
        c_in,
        c_context,
        d_model,
        embed_type="timeF",
        freq="h",
        dropout=0.1,
        d_x_embedding_router=None,
    ):
        super(ContextEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.context_embedding = TokenEmbedding(c_in=c_context, d_model=d_model)

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.d_x_embedding_router = d_x_embedding_router

    def forward(self, x, x_mark):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, L, c_in).
        x_mark : torch.Tensor
            Context and temporal features of shape (B, L, D_C).

        Returns
        -------
        torch.Tensor
            Embedded output of shape (B, L, d_model).
        """
        _, _, D_C = x_mark.shape
        # time_features_mask = torch.tensor([0]*(D_C-4)+[1]*4)
        time_features_mask = self.d_x_embedding_router

        x = self.value_embedding(x) + self.position_embedding(x)

        if time_features_mask is not None:
            # flip to month, day, weekday, hour
            time_features = x_mark[..., time_features_mask.bool()].flip(2)
            context_features = x_mark[..., ~time_features_mask.bool()]
            x += self.context_embedding(context_features) + self.temporal_embedding(
                time_features
            )
        else:
            x += self.context_embedding(x_mark)

        return self.dropout(x)


# used for ablation study
class ContextEmbedding_wo_context(nn.Module):
    def __init__(
        self,
        c_in,
        c_context,
        d_model,
        embed_type="timeF",
        freq="h",
        dropout=0.1,
        d_x_embedding_router=None,
    ):
        super(ContextEmbedding_wo_context, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.context_embedding = TokenEmbedding(c_in=c_context, d_model=d_model)

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.d_x_embedding_router = d_x_embedding_router

    def forward(self, x, x_mark):
        _, _, D_C = x_mark.shape
        # time_features_mask = torch.tensor([0]*(D_C-4)+[1]*4)
        time_features_mask = self.d_x_embedding_router
        # flip to month, day, weekday, hour
        time_features = x_mark[..., time_features_mask.bool()].flip(2)
        # context_features = x_mark[...,~time_features_mask.bool()]

        x = (
            self.value_embedding(x)
            + self.temporal_embedding(time_features)
            + self.position_embedding(x)
        )
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
