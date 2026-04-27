"""
Data dropout and masking utilities for Spacetimeformer encoder embedding.

This module provides dropout strategies applied to the input sequence before
embedding, used to regularize reconstruction during training. It includes
ReconstructionDropout (combining full-timestep, element-wise, and subsequence
masking), RandomMask (replacing values with a constant), and the low-level
create_subsequence_mask helper.

See Also
--------
ReconstructionDropout : Main dropout combining multiple masking strategies.
RandomMask : Replaces masked values with a constant.
create_subsequence_mask : Low-level helper for subsequence masking.

Examples
--------
>>> from picid.model.forecasters.spacetimeformer_model.nn.data_dropout import (
...     ReconstructionDropout,
...     RandomMask,
... )
>>> dropout = ReconstructionDropout(drop_standard=0.1)
>>> masked = RandomMask(prob=0.1, change_to_val=0.0)
"""

import random

import torch
from torch import nn
from torch.distributions.geometric import Geometric
from torch.distributions.binomial import Binomial


def create_subsequence_mask(o, r=0.15, lm=3, stateful=True, sync=False):
    """
    Create a boolean mask over random subsequences of the input tensor.

    Masks contiguous subsequences rather than individual elements, encouraging
    the model to learn from partial sequences. Uses Geometric/Binomial
    distributions (borrowed from IBM CodeFlare). When stateful=True, alternates
    between masked and unmasked runs with geometrically distributed lengths.

    Parameters
    ----------
    o : torch.Tensor
        Input tensor of shape (mask_dims, mask_len) or (n_masks, mask_dims, mask_len).
        Will be expanded to 3D if 2D.
    r : float, optional
        Target fraction of elements to mask (default 0.15).
    lm : int, optional
        Mean length of masked/unmasked runs when stateful (default 3).
    stateful : bool, optional
        If True, use Geometric distribution for structured runs; else Bernoulli
        per element (default True).
    sync : bool or str, optional
        If True or "random", apply the same mask across all dimensions (default False).

    Returns
    -------
    torch.Tensor
        Boolean mask of same shape as input (after 3D expansion). True = keep, False = mask.
    """
    if r <= 0:
        return torch.zeros_like(o).bool()
    device = o.device
    if o.ndim == 2:
        o = o[None]
    n_masks, mask_dims, mask_len = o.shape
    if sync == "random":
        sync = random.random() > 0.5
    dims = 1 if sync else mask_dims
    if stateful:
        numels = n_masks * dims * mask_len
        pm = torch.tensor([1 / lm], device=device)
        pu = torch.clip(pm * (r / max(1e-6, 1 - r)), 1e-3, 1)
        zot, proba_a, proba_b = (
            (torch.as_tensor([False, True], device=device), pu, pm)
            if random.random() > pm
            else (torch.as_tensor([True, False], device=device), pm, pu)
        )
        max_len = max(
            1,
            2
            * torch.div(numels, (1 / pm + 1 / pu), rounding_mode="floor").long().item(),
        )
        for i in range(10):
            _dist_a = (Geometric(probs=proba_a).sample([max_len]) + 1).long()
            _dist_b = (Geometric(probs=proba_b).sample([max_len]) + 1).long()
            dist_a = _dist_a if i == 0 else torch.cat((dist_a, _dist_a), dim=0)  # noqa
            dist_b = _dist_b if i == 0 else torch.cat((dist_b, _dist_b), dim=0)  # noqa
            add = torch.add(dist_a, dist_b)
            if torch.gt(torch.sum(add), numels):
                break
        dist_len = torch.argmax((torch.cumsum(add, 0) >= numels).float()) + 1
        if dist_len % 2:
            dist_len += 1
        repeats = torch.cat((dist_a[:dist_len], dist_b[:dist_len]), -1).flatten()
        zot = zot.repeat(dist_len)
        mask = torch.repeat_interleave(zot, repeats)[:numels].reshape(
            n_masks, dims, mask_len
        )
    else:
        probs = torch.tensor(r, device=device)
        mask = Binomial(1, probs).sample((n_masks, dims, mask_len)).bool()
    if sync:
        mask = mask.repeat(1, mask_dims, 1)
    return mask


class ReconstructionDropout(nn.Module):
    """
    Dropout applied to encoder input for reconstruction regularization.

    Combines three masking strategies during training: (1) full timestep dropout,
    (2) element-wise standard dropout, and (3) subsequence masking via
    create_subsequence_mask. Optionally skips all dropout with probability
    skip_all_drop so the model sees unmasked inputs during training, improving
    test-time behavior when no dropout is applied.

    Used by Spacetimeformer's encoder embedding (see model.py recon_mask_* args).

    Parameters
    ----------
    drop_full_timesteps : float, optional
        Probability of dropping entire timesteps (default 0.0).
    drop_standard : float, optional
        Element-wise dropout probability (default 0.0).
    drop_seq : float, optional
        Subsequence mask rate passed to create_subsequence_mask (default 0.0).
    drop_max_seq_len : int, optional
        Mean run length for subsequence masking (default 5).
    skip_all_drop : float, optional
        Probability of applying no dropout; 1.0 = always apply (default 1.0).
    """

    def __init__(
        self,
        drop_full_timesteps=0.0,
        drop_standard=0.0,
        drop_seq=0.0,
        drop_max_seq_len=5,
        skip_all_drop=1.0,
    ):
        """
        Initialize ReconstructionDropout.

        Parameters
        ----------
        drop_full_timesteps : float, optional
            Probability of dropping entire timesteps (default 0.0).
        drop_standard : float, optional
            Element-wise dropout probability (default 0.0).
        drop_seq : float, optional
            Subsequence mask rate passed to create_subsequence_mask (default 0.0).
        drop_max_seq_len : int, optional
            Mean run length for subsequence masking (default 5).
        skip_all_drop : float, optional
            Probability of applying no dropout; 1.0 = always apply (default 1.0).
        """
        super().__init__()
        self.drop_full_timesteps = drop_full_timesteps
        self.drop_standard = drop_standard
        self.drop_seq = drop_seq
        self.drop_max_seq_len = drop_max_seq_len
        self.skip_all_drop = skip_all_drop

    def forward(self, y):
        """
        Apply dropout mask to input during training.

        Parameters
        ----------
        y : torch.Tensor
            Input of shape (batch_size, length, dim).

        Returns
        -------
        torch.Tensor
            Masked input (same shape); unchanged if eval mode or skip_all_drop=1.0.
        """
        bs, length, dim = y.shape
        dev = y.device

        if self.training and self.skip_all_drop < 1.0:
            # mask full timesteps
            full_timestep_mask = torch.bernoulli(
                (1.0 - self.drop_full_timesteps) * torch.ones(bs, length, 1)
            ).to(dev)

            # mask each element indp
            standard_mask = torch.bernoulli(
                (1.0 - self.drop_standard) * torch.ones(bs, length, dim)
            ).to(dev)

            # subsequence mask
            seq_mask = (
                1.0
                - create_subsequence_mask(
                    y.transpose(1, 2), r=self.drop_seq, lm=self.drop_max_seq_len
                )
                .transpose(1, 2)
                .float()
            )

            # skip all dropout occasionally so when there is no dropout
            # at test time the model has seen that before. (I am not sure
            # the usual activation strength adjustment makes sense here)
            skip_all_drop_mask = torch.bernoulli(
                1.0 - self.skip_all_drop * torch.ones(bs, 1, 1)
            ).to(dev)

            mask = 1.0 - (
                (1.0 - (full_timestep_mask * standard_mask * seq_mask))
                * skip_all_drop_mask
            )

            return y * mask
        else:
            return y

    def __repr__(self):  # numpydoc ignore=RT01
        """Return string representation of dropout configuration."""
        return f"Timesteps {self.drop_full_timesteps}, Standard {self.drop_standard}, Seq (max len = {self.drop_max_seq_len}) {self.drop_seq}, Skip All Drop {self.skip_all_drop}"


class RandomMask(nn.Module):
    """
    Replace input values with a constant at random positions during training.

    Applies Bernoulli dropout per timestep (broadcast across features) and
    replaces masked positions with change_to_val. No-op when change_to_val is
    None or in eval mode.

    Parameters
    ----------
    prob : float
        Probability of keeping each timestep; 1 - prob = mask rate.
    change_to_val : float or None
        Value to use for masked positions. If None, forward is identity.
    """

    def __init__(self, prob, change_to_val):
        """
        Initialize RandomMask.

        Parameters
        ----------
        prob : float
            Probability of keeping each timestep; 1 - prob = mask rate.
        change_to_val : float or None
            Value to use for masked positions. If None, forward is identity.
        """
        super().__init__()
        self.prob = prob
        self.change_to_val = change_to_val

    def forward(self, y):
        """
        Replace random timesteps with change_to_val during training.

        Parameters
        ----------
        y : torch.Tensor
            Input of shape (batch_size, length, dim).

        Returns
        -------
        torch.Tensor
            Masked input; unchanged if eval mode or change_to_val is None.
        """
        bs, length, dy = y.shape
        if not self.training or self.change_to_val is None:
            return y
        mask = torch.bernoulli((1.0 - self.prob) * torch.ones(bs, length, 1))
        mask.requires_grad = False
        mask = mask.to(y.device)
        masked_y = (y * mask) + (self.change_to_val * (1.0 - mask))
        return masked_y

    def __repr__(self):  # numpydoc ignore=RT01
        """Return string representation of mask configuration."""
        return f"RandomMask(prob = {self.prob}, val = {self.change_to_val})"
