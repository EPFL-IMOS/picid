from __future__ import annotations

import random
import numpy as np

import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, X):
        return X + torch.normal(torch.zeros_like(X), self.std)


class TimeSampler:
    """Input sampler for generating features for prior datasets.

    Creates a monotonically increasing sequence of features with optional noise.
    The monotonic sequence is of the form x**k where x is a uniform sequence between [0, 1] and k is a random exponent.
    The noise is added as Gaussian noise with a specified standard deviation.

    Parameters
    ----------
    seq_len : int
        Length of sequence to generate

    num_features : int
        Number of features to generate

    exponent  : str, default='mixed'
        Feature sampling strategy ('normal', 'mixed', 'uniform')

    noise_std : float, default=0.0
        Standard deviation of Gaussian noise to add to the features

    device : str, default='cpu'
        Device to store tensors on
    """

    def __init__(self, seq_len, num_features, exponent = 1, noise_std = 0.01, device="cpu"):
        self.seq_len = seq_len
        self.num_features = 1 # Only one feature is generated as it is a time series
        self.exponent = exponent
        self.noise_std = noise_std
        self.device = device


    def sample(self, return_numpy=False):
        """Generate features according to the specified sampling strategy.

        Returns
        -------
        X : torch.Tensor
            Generated features of shape (seq_len, num_features)
        """
        x = torch.linspace(0, 1, self.seq_len, device=self.device)
        x = x ** self.exponent  # Apply the exponent to create a monotonically increasing sequence
        x = x + torch.normal(0, self.noise_std, size=x.shape, device=self.device)  # Add Gaussian noise
        X = x.unsqueeze(-1)  # Add feature dimension

        return X.cpu().numpy() if return_numpy else X


class XSampler:
    """Input sampler for generating features for prior datasets.

    Supports multiple feature distribution types:
    - Normal: Standard normal distribution
    - Multinomial: Categorical features with random number of categories
    - Zipf: Power law distributed features
    - Mixed: Random combination of the above

    Parameters
    ----------
    seq_len : int
        Length of sequence to generate

    num_features : int
        Number of features to generate

    pre_stats : bool
        Whether to pre-generate statistics for the input features

    sampling : str, default='mixed'
        Feature sampling strategy ('normal', 'mixed', 'uniform', time, 'time_and_uniform', 'time_and_normal')

    device : str, default='cpu'
        Device to store tensors on
    """

    def __init__(self, seq_len, num_features, pre_stats=False, sampling="mixed", time_exponent=1., device="cpu"):
        self.seq_len = seq_len
        self.num_features = num_features
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device
        self.exponent = time_exponent

        if pre_stats:
            self._pre_stats()

    def _pre_stats(self):
        means = np.random.normal(0, 1, self.num_features)
        stds = np.abs(np.random.normal(0, 1, self.num_features) * means)
        self.means = torch.tensor(means, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.seq_len, 1)
        self.stds = torch.tensor(stds, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.seq_len, 1)

    def sample(self, return_numpy=False):
        """Generate features according to the specified sampling strategy.

        Returns
        -------
        X : torch.Tensor
            Generated features of shape (seq_len, num_features)
        """
        samplers = {"normal": self.sample_normal_all, 
                    "mixed": self.sample_mixed, 
                    "uniform": self.sample_uniform,
                    "time": self.sample_time,
                    "time_and_uniform": self.sample_time_and_uniform,
                    "time_and_normal": self.sample_time_and_normal,
                    "time_multiscale": self.sample_time_multiscale,}
        if self.sampling not in samplers:
            raise ValueError(f"Invalid sampling method: {self.sampling}")
        X = samplers[self.sampling]()

        return X.cpu().numpy() if return_numpy else X

    def sample_normal_all(self):
        if self.pre_stats:
            X = torch.normal(self.means, self.stds.abs()).float()
        else:
            X = torch.normal(0.0, 1.0, (self.seq_len, self.num_features), device=self.device).float()
        return X

    def sample_uniform(self):
        """Generate uniformly distributed features."""
        return torch.rand((self.seq_len, self.num_features), device=self.device)

    def sample_normal(self, n=None):
        """Generate normally distributed features.

        Parameters
        ----------
        n : int
            Index of the feature to generate
        """

        if self.pre_stats:
            return torch.normal(self.means[:, n], self.stds[:, n].abs()).float()
        else:
            return torch.normal(0.0, 1.0, (self.seq_len,), device=self.device).float()

    def sample_multinomial(self):
        """Generate categorical features."""
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.seq_len, replacement=True)
        x = x.float()
        return (x - x.mean()) / x.std()

    def sample_zipf(self):
        """Generate Zipf-distributed features."""
        x = np.random.zipf(2.0 + random.random() * 2, (self.seq_len,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def sample_mixed(self):
        """Generate features with mixed distributions."""
        X = []
        zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
        for n in range(self.num_features):
            if random.random() > normal_p:
                x = self.sample_normal(n)
            elif random.random() > multi_p:
                x = self.sample_multinomial()
            elif random.random() > zipf_p:
                x = self.sample_zipf()
            else:
                x = torch.rand((self.seq_len,), device=self.device)
            X.append(x)
        return torch.stack(X, -1)
    
    def sample_time(self):
        x = torch.linspace(0, 1, self.seq_len, device=self.device).unsqueeze(-1)
        x = x ** self.exponent  # Apply the exponent to create a monotonically increasing sequence
        noise = 0*torch.rand((self.seq_len, self.num_features-1), device=self.device)
        X = torch.cat([x, noise], dim=-1)
        return X.float()  
    
    def sample_time_and_uniform(self):
        x = torch.linspace(0, 1, self.seq_len, device=self.device).unsqueeze(-1)
        x = x ** self.exponent  # Apply the exponent to create a monotonically increasing sequence
        noise = torch.rand((self.seq_len, self.num_features-1), device=self.device)
        X = torch.cat([x, noise], dim=-1)
        return X.float()  

    def sample_time_and_normal(self):
        x = torch.linspace(0, 1, self.seq_len, device=self.device).unsqueeze(-1)
        x = x ** self.exponent
        if self.pre_stats:
            noise = torch.normal(self.means, self.stds.abs())[:,-1].float()
        else:
            noise = torch.normal(0.0, 1.0, (self.seq_len, self.num_features-1), device=self.device).float()
        return torch.cat([x, noise], dim=-1).float()

    def sample_time_multiscale(self):
        log_scales = torch.randn(self.num_features, device=self.device)*2
        scales = torch.exp(log_scales).unsqueeze(0)
        x = torch.linspace(0, 1, self.seq_len, device=self.device).unsqueeze(-1)
        x_scales = x * scales  # Scale the time sequence
        return x_scales.float()
