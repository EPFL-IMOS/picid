import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feed-forward network.

    Parameters
    ----------
    attention : nn.Module
        Attention module (e.g., AttentionLayer wrapping FullAttention).
    d_model : int
        Model dimension.
    d_ff : int, optional
        Feed-forward hidden dimension. Defaults to 4 * d_model.
    dropout : float, optional
        Dropout probability. Defaults to 0.1.
    activation : str, optional
        Activation function: "relu" or "gelu". Defaults to "relu".
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, D).
        attn_mask : object, optional
            Optional attention mask.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, D).
        object
            Attention weights (or None if output_attention=False).
        """
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    Stack of encoder layers with optional conv layers and normalization.

    Parameters
    ----------
    attn_layers : list of nn.Module
        List of EncoderLayer modules.
    conv_layers : list of nn.Module, optional
        Optional conv layers between attention layers.
    norm_layer : nn.Module, optional
        Optional final normalization layer (e.g., LayerNorm).
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, D).
        attn_mask : object, optional
            Optional attention mask.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, D).
        list
            List of attention weights from each layer.
        """
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention, cross-attention, and feed-forward.

    Parameters
    ----------
    self_attention : nn.Module
        Self-attention module (masked for autoregressive decoding).
    cross_attention : nn.Module
        Cross-attention module (attends to encoder output).
    d_model : int
        Model dimension.
    d_ff : int, optional
        Feed-forward hidden dimension. Defaults to 4 * d_model.
    dropout : float, optional
        Dropout probability. Defaults to 0.1.
    activation : str, optional
        Activation function: "relu" or "gelu". Defaults to "relu".
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Decoder input of shape (B, L, D).
        cross : torch.Tensor
            Encoder output of shape (B, S, D).
        x_mask : object, optional
            Optional mask for self-attention.
        cross_mask : object, optional
            Optional mask for cross-attention.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, D).
        """
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """
    Stack of decoder layers with optional normalization and projection.

    Parameters
    ----------
    layers : list of nn.Module
        List of DecoderLayer modules.
    norm_layer : nn.Module, optional
        Optional final normalization layer.
    projection : nn.Module, optional
        Optional output projection layer.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Decoder input of shape (B, L, D).
        cross : torch.Tensor
            Encoder output of shape (B, S, D).
        x_mask : object, optional
            Optional mask for self-attention.
        cross_mask : object, optional
            Optional mask for cross-attention.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, D).
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
