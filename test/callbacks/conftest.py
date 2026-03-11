"""
Shared fixtures for callbacks tests.

Provides PHM-realistic mock trainer, pl_module, and batch structures
for testing ModelCheckpointWithConfig, ResourceTracker, PipelineTimer.
Docs: dataobject.md, datasets.md (batch shapes).
"""

# Mock pynvml before resource_tracker is imported (avoids NVML init on machines without GPU)
import sys
from unittest.mock import MagicMock

_mock_pynvml = MagicMock()
_mock_pynvml.nvmlInit = MagicMock()
sys.modules["pynvml"] = _mock_pynvml

import tempfile

import pytest
import torch


@pytest.fixture
def phm_batch_features():
    """
    Batch with "features" key for ResourceTracker._extract_features.
    Doc: dataobject.md - batch contains features/target.
    """
    return {"features": torch.randn(4, 20, 5), "rul": torch.randn(4, 1)}


@pytest.fixture
def phm_batch_context_x():
    """Batch with context.x for RUL/context datasets."""
    return {"context": {"x": torch.randn(4, 10, 3)}, "rul": torch.randn(4, 1)}


@pytest.fixture
def phm_batch_other_tensor():
    """Batch with no 'features' or context.x; first tensor (not 'rul') is used."""
    return {"other": torch.randn(2, 5), "rul": torch.randn(2, 1)}


@pytest.fixture
def phm_batch_list():
    """Batch as list (e.g. [features_tensor, target_tensor])."""
    return [torch.randn(4, 10), torch.randn(4, 1)]


@pytest.fixture
def mock_trainer_global_zero():
    """Trainer mock with is_global_zero=True for main process."""
    t = MagicMock()
    t.is_global_zero = True
    t.print = MagicMock()
    t.logger = MagicMock()
    t.logger.log_metrics = MagicMock()
    return t


@pytest.fixture
def mock_trainer_not_global_zero():
    """Trainer mock with is_global_zero=False."""
    t = MagicMock()
    t.is_global_zero = False
    return t


@pytest.fixture
def mock_pl_module():
    """Minimal Lightning module mock (global_rank, log, log_dict)."""
    m = MagicMock()
    m.global_rank = 0
    m.log = MagicMock()
    m.log_dict = MagicMock()
    return m


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary directory for checkpoint/config file tests."""
    with tempfile.TemporaryDirectory() as d:
        yield d
