"""
Shared fixtures for picid/data test suite.

Provides synthetic but representative data (numpy, torch, small arrays) for
high-fidelity pipeline testing without mocks. Use these fixtures for
data integrity, shape consistency, and collation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from test.fixtures.builders import (
    make_fault_classification_targets_continuing_features,
    make_forecasting_target_continuing_features,
    make_synthetic_features_float32,
    make_synthetic_fit_predict_3d,
    make_synthetic_image_like_uint8,
    make_synthetic_normalized_float32,
)

# Optional: for ragged/sliding-window tests
try:
    import awkward as ak
    from picid.utils.awkward_utils import ak_regularize_regular_axes

    HAS_AWKWARD = True
except ImportError:
    HAS_AWKWARD = False


# ---------------------------------------------------------------------------
# Seeds & reproducibility
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_seed():
    """Set a fixed seed for reproducible synthetic data and transforms."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ---------------------------------------------------------------------------
# Synthetic data: dense time series (representative of RUL/forecasting)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_features_float32(fixed_seed):
    """Dense float32 feature matrix: (T, F). Values in [0, 1] for normalized pipelines."""
    return make_synthetic_features_float32(
        seed=fixed_seed, time_steps=100, n_features=8
    )


@pytest.fixture
def synthetic_target_rul(synthetic_features_float32):
    """RUL target aligned with feature length: counts down from T-1 to 0."""
    T = synthetic_features_float32.shape[0]
    return np.arange(T - 1, -1, -1, dtype=np.float32).reshape(-1, 1)


@pytest.fixture
def synthetic_data_dict_rul(synthetic_features_float32, synthetic_target_rul):
    """Minimal data_dict for RUL task: features + rul."""
    return {
        "features": synthetic_features_float32,
        "rul": synthetic_target_rul,
    }


@pytest.fixture
def synthetic_data_dict_forecasting(synthetic_features_float32, fixed_seed):
    """Data dict for forecasting task: features + target (same length)."""
    target = make_forecasting_target_continuing_features(
        synthetic_features_float32, seed=fixed_seed
    )
    return {"features": synthetic_features_float32, "target": target}


@pytest.fixture
def synthetic_data_dict_fault_classification(synthetic_features_float32, fixed_seed):
    """Data dict for fault_classification: features + fault_class (int-like float)."""
    fault_class = make_fault_classification_targets_continuing_features(
        synthetic_features_float32, seed=fixed_seed, n_classes=5
    )
    return {"features": synthetic_features_float32, "fault_classification": fault_class}


# ---------------------------------------------------------------------------
# Multi-unit data (lists of arrays) for HydraConcatDataset-style tests
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_multi_unit_rul(synthetic_data_dict_rul):
    """List of 3 units, each with features and rul (for Hydra init)."""
    feats = synthetic_data_dict_rul["features"]
    rul = synthetic_data_dict_rul["rul"]
    n_units = 3
    # Same length per unit for simplicity
    T_unit = feats.shape[0] // n_units
    features_list = [feats[i * T_unit : (i + 1) * T_unit] for i in range(n_units)]
    rul_list = [rul[i * T_unit : (i + 1) * T_unit] for i in range(n_units)]
    return {"features": features_list, "rul": rul_list}


# ---------------------------------------------------------------------------
# FitPredict 3D arrays (n_tasks, n_samples, n_features)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_fit_predict_3d(fixed_seed):
    """X: (n_tasks, n_samples, n_features), y: (n_tasks, n_samples, n_targets)."""
    return make_synthetic_fit_predict_3d(seed=fixed_seed)


# ---------------------------------------------------------------------------
# Value-range fixtures (normalized [0,1] vs [0,255])
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_image_like_uint8(fixed_seed):
    """Synthetic 'image' data in uint8 [0, 255] for dtype/range tests."""
    return make_synthetic_image_like_uint8(seed=fixed_seed)


@pytest.fixture
def synthetic_normalized_float32(fixed_seed):
    """Float32 in [0, 1] for normalized pipeline assertions."""
    return make_synthetic_normalized_float32(seed=fixed_seed)


# ---------------------------------------------------------------------------
# Ragged / awkward (optional)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_ragged_two_units():
    """Two units with different lengths for ragged sequencer tests."""
    if not HAS_AWKWARD:
        pytest.skip("awkward not available")
    c0 = [[x * 0.1, x * 0.2] for x in range(15)]
    c1 = [[x * 0.1, x * 0.2] for x in range(25)]
    arr = ak_regularize_regular_axes(ak.Array([c0, c1]))
    return arr


# ---------------------------------------------------------------------------
# Batch sizes for parametrized collation tests
# ---------------------------------------------------------------------------

BATCH_SIZES = [1, 2, 4, 8, 16]


@pytest.fixture(params=BATCH_SIZES)
def batch_size(request):
    return request.param
