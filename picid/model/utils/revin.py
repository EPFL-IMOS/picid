"""Reversible Instance Normalization (RevIN) from https://github.com/ts-kim/RevIN."""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Moving average over the time dimension using 1D average pooling.

    Applies padding on both ends of the time series before pooling to preserve
    output length. Expects input shape (batch, seq_len, channels).

    Parameters
    ----------
    kernel_size : int
        Size of the averaging window.
    stride : int
        Stride for the average pooling operation.
    """

    def __init__(self, kernel_size, stride):
        """
        Initialize MovingAvg.

        Parameters
        ----------
        kernel_size : int
            Size of the averaging window.
        stride : int
            Stride for the average pooling operation.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        Apply moving average along the time dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, channels).

        Returns
        -------
        torch.Tensor
            Smoothed tensor of shape (batch, seq_len, channels).
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """
    Decompose a time series into residual and trend components.

    Trend is computed via moving average; residual is the remainder.

    Parameters
    ----------
    kernel_size : int
        Size of the moving average kernel for trend extraction.
    """

    def __init__(self, kernel_size):
        """
        Initialize SeriesDecomposition.

        Parameters
        ----------
        kernel_size : int
            Size of the moving average kernel for trend extraction.
        """
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        """
        Decompose input into residual and trend.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, channels).

        Returns
        -------
        res : torch.Tensor
            Residual component, same shape as input.
        trend : torch.Tensor
            Trend component (moving mean), same shape as input.
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.

    Normalizes each channel independently using instance statistics.
    Supports reversible denormalization for forecasting.

    Parameters
    ----------
    num_features : int
        The number of features or channels.
    eps : float, optional
        A value added for numerical stability (default: 1e-5).
    affine : bool, optional
        If True, RevIN has learnable affine parameters (default: True).
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Initialize RevIN.

        Parameters
        ----------
        num_features : int
            The number of features or channels.
        eps : float, optional
            A value added for numerical stability (default: 1e-5).
        affine : bool, optional
            If True, RevIN has learnable affine parameters (default: True).
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, update_stats=True):
        """
        Normalize or denormalize the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, num_features).
        mode : str
            Either "norm" (normalize) or "denorm" (denormalize).
        update_stats : bool, optional
            If True, recompute mean/std from x (default: True).
            Only used when mode is "norm".

        Returns
        -------
        torch.Tensor
            Normalized or denormalized tensor, same shape as input.

        Raises
        ------
        AssertionError
            If input does not have 3 dimensions.
        NotImplementedError
            If mode is not "norm" or "denorm".
        """
        assert x.ndim == 3
        if mode == "norm":
            if update_stats:
                self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        """Initialize learnable affine parameters (weight and bias)."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Compute mean and stdev over time dimension for normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, num_features).

        Returns
        -------
        None
            Updates self.mean and self.stdev in place.
        """
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        """
        Apply normalization using stored mean and stdev.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize.

        Returns
        -------
        torch.Tensor
            Normalized tensor, same shape as input.
        """
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        Reverse normalization to restore original scale.

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor to denormalize.

        Returns
        -------
        torch.Tensor
            Denormalized tensor, same shape as input.
        """
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
