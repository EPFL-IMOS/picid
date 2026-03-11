"""Shared fixtures for optimizer tests.

Provides PHM-relevant mock model parameters and optimizer configurations
aligned with docs/dataobject.md and docs/datasets.md (RUL/forecasting pipelines).
"""

import pytest
import torch.nn as nn


# -----------------------------------------------------------------------------
# Gold-standard: nominal PHM model parameters (healthy training setup)
# Docs: dataobject.md - model parameters are typically nn.Module.parameters()
# -----------------------------------------------------------------------------


@pytest.fixture
def phm_model_parameters():
    """
    Realistic model parameters for PHM pipeline testing.

    **PHM Logic**: Simulates a small RUL/forecasting backbone (e.g. linear + LSTM-like)
    with parameter count and structure typical of tabular/time-series PHM models.

    **Methodology**: A tiny nn.Module is used so that torch.optim can accept
    params=model.parameters() as in TrainingLightningModule.configure_optimizers().

    **Expected outcome**: Iterable of tensors with requires_grad=True for optimizer.
    """
    model = nn.Sequential(
        nn.Linear(5, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    return list(model.parameters())


@pytest.fixture
def phm_model_parameters_single():
    """
    Single-parameter-group model for edge-case optimizer configuration.

    **PHM Logic**: Some baselines use a single linear layer; optimizer must
    handle a single parameter list.

    **Expected outcome**: One group of parameters.
    """
    layer = nn.Linear(3, 1)
    return list(layer.parameters())
