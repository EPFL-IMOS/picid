import torch
import torch.nn as nn
from .cross_attn import TwoStageAttentionLayer
from math import ceil


class SegMerging(nn.Module):
    """
    Merge adjacent segments to build a coarser Crossformer scale.

    Parameters
    ----------
    d_model : int
        Embedding dimension of each segment.
    win_size : int
        Number of adjacent segments merged into one coarser segment.
    norm_layer : callable, optional
        Normalization layer applied before the linear projection.
    """

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        Merge the segment axis while preserving batch and channel axes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, ts_d, seg_num, d_model]``.

        Returns
        -------
        torch.Tensor
            Tensor with merged segments.
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i :: self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    """
    Apply optional merging and then a stack of two-stage attention layers.

    Parameters
    ----------
    win_size : int
        Merge window size. A value of ``1`` disables merging.
    d_model : int
        Hidden dimension used by the attention stack.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward hidden dimension.
    depth : int
        Number of two-stage attention layers.
    dropout : float
        Dropout probability.
    seg_num : int, optional
        Number of input segments at this scale.
    factor : int, optional
        Router width multiplier used by the attention block.
    """

    def __init__(
        self, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, factor=10
    ):
        super(scale_block, self).__init__()

        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout)
            )

    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    """
    Stack Crossformer scale blocks into the encoder tower.

    Parameters
    ----------
    e_blocks : int
        Number of encoder scales.
    win_size : int
        Merge window size for the coarser scales.
    d_model : int
        Hidden dimension used by the attention stack.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward hidden dimension.
    block_depth : int
        Number of two-stage attention layers per scale.
    dropout : float
        Dropout probability.
    in_seg_num : int, optional
        Number of input segments at the finest scale.
    factor : int, optional
        Router width multiplier used by the attention block.
    """

    def __init__(
        self,
        e_blocks,
        win_size,
        d_model,
        n_heads,
        d_ff,
        block_depth,
        dropout,
        in_seg_num=10,
        factor=10,
    ):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(
            scale_block(
                1, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num, factor
            )
        )
        for i in range(1, e_blocks):
            self.encode_blocks.append(
                scale_block(
                    win_size,
                    d_model,
                    n_heads,
                    d_ff,
                    block_depth,
                    dropout,
                    ceil(in_seg_num / win_size**i),
                    factor,
                )
            )

    def forward(self, x):
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x
