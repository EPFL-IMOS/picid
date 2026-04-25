import torch.nn as nn
from einops import rearrange
from .cross_attn import AttentionLayer, TwoStageAttentionLayer


class DecoderLayer(nn.Module):
    """
    Decode one Crossformer scale and emit a scale-specific prediction.

    Parameters
    ----------
    seg_len : int
        Length of the segment predicted by this layer.
    d_model : int
        Hidden dimension of the decoder state.
    n_heads : int
        Number of attention heads.
    d_ff : int, optional
        Feed-forward hidden width.
    dropout : float, optional
        Dropout probability.
    out_seg_num : int, optional
        Number of output segments processed by the self-attention block.
    factor : int, optional
        Router width multiplier used by the two-stage attention block.
    """

    def __init__(
        self,
        seg_len,
        d_model,
        n_heads,
        d_ff=None,
        dropout=0.1,
        out_seg_num=10,
        factor=10,
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(
            out_seg_num, factor, d_model, n_heads, d_ff, dropout
        )
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        """
        Decode one scale using the previous decoder state and encoder skip.

        Parameters
        ----------
        x : torch.Tensor
            Decoder state from the previous layer.
        cross : torch.Tensor
            Encoder features for the same scale.

        Returns
        -------
        tuple of torch.Tensor
            Updated decoder state and the scale prediction.
        """

        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, "b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model")

        cross = rearrange(
            cross, "b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model"
        )
        tmp = self.cross_attention(
            x,
            cross,
            cross,
        )
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        dec_output = rearrange(
            dec_output,
            "(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model",
            b=batch,
        )
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(
            layer_predict, "b out_d seg_num seg_len -> b (out_d seg_num) seg_len"
        )

        return dec_output, layer_predict


class Decoder(nn.Module):
    """
    Aggregate the scale-wise decoder predictions into the final output.

    Parameters
    ----------
    seg_len : int
        Length of the segment predicted by each decoder layer.
    d_layers : int
        Number of stacked decoder layers.
    d_model : int
        Hidden dimension of the decoder state.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward hidden width.
    dropout : float
        Dropout probability.
    router : bool, optional
        Compatibility flag kept from the upstream implementation.
    out_seg_num : int, optional
        Number of output segments processed by the self-attention block.
    factor : int, optional
        Router width multiplier used by the two-stage attention block.
    """

    def __init__(
        self,
        seg_len,
        d_layers,
        d_model,
        n_heads,
        d_ff,
        dropout,
        router=False,
        out_seg_num=10,
        factor=10,
    ):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(
                DecoderLayer(
                    seg_len, d_model, n_heads, d_ff, dropout, out_seg_num, factor
                )
            )

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(
            final_predict,
            "b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d",
            out_d=ts_d,
        )

        return final_predict
