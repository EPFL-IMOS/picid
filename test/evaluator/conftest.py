"""
Shared test fixtures for evaluator tests.

This module provides comprehensive, realistic data fixtures for testing
all evaluator types. The fixtures are designed to be "interesting" and
challenging, testing edge cases, realistic PHM scenarios, and various
data distributions.

PHM Context:
- RUL predictions: Normalized [0, 1] representing remaining useful life
- Classification: Fault detection with multiple classes
- Forecasting: Multi-step ahead predictions
- Multi-unit: Equipment with different degradation patterns
"""

import pytest
import numpy as np
from typing import Dict, Any, Optional

from test.fixtures.builders import make_regression_batch


# =============================================================================
# === REGRESSION/FORECASTING DATA FIXTURES ===
# =============================================================================


@pytest.fixture
def regression_data_basic():
    """
    Basic regression data: Simple RUL predictions.

    Shape: (batch=5, time=1, features=1)
    Values: Linear degradation from 0.9 to 0.1

    **Interesting aspects**:
    - Small batch size
    - Single timestep (typical RUL prediction)
    - Values span full normalized range [0, 1]
    """
    return make_regression_batch(seed=42, batch_size=5)


@pytest.fixture
def regression_data_large_batch():
    """
    Large batch regression data: Many samples for stress testing.

    Shape: (batch=1000, time=1, features=1)
    Values: Realistic RUL distribution with noise

    **Interesting aspects**:
    - Large batch size (tests performance, memory)
    - Realistic noise distribution
    - Some outliers (faulty predictions)
    """
    batch_size = 1000
    rng = np.random.RandomState(123)

    # Main distribution: healthy equipment (high RUL)
    healthy_rul = rng.beta(5, 2, size=int(batch_size * 0.7))  # Skewed high

    # Degraded equipment (low RUL)
    degraded_rul = rng.beta(2, 5, size=int(batch_size * 0.25))  # Skewed low

    # Faulty predictions (outliers)
    faulty_rul = rng.uniform(0.0, 1.0, size=int(batch_size * 0.05))

    all_rul = np.concatenate([healthy_rul, degraded_rul, faulty_rul])
    rng.shuffle(all_rul)

    preds = all_rul[:batch_size].reshape(batch_size, 1, 1)
    targets = preds + rng.randn(batch_size, 1, 1) * 0.1
    targets = np.clip(targets, 0.0, 1.0)

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
    }


@pytest.fixture
def regression_data_single_sample():
    """
    Single sample regression data: Edge case.

    Shape: (batch=1, time=1, features=1)

    **Interesting aspects**:
    - Minimum batch size
    - Tests edge case handling
    """
    return {
        "predictions": np.array([[[0.5]]], dtype=np.float32),
        "targets": np.array([[[0.52]]], dtype=np.float32),
    }


@pytest.fixture
def forecasting_data_multi_step():
    """
    Multi-step forecasting data: Time series predictions.

    Shape: (batch=10, time=24, features=1)
    Values: Degradation curve over 24 timesteps

    **Interesting aspects**:
    - Multiple timesteps (forecasting horizon)
    - Realistic degradation trend (exponential decay)
    - Some units degrade faster than others
    """
    batch_size = 10
    horizon = 24
    rng = np.random.RandomState(456)

    preds = np.zeros((batch_size, horizon, 1))
    targets = np.zeros((batch_size, horizon, 1))

    for i in range(batch_size):
        # Each unit has different degradation rate
        decay_rate = rng.uniform(0.02, 0.05)
        initial_rul = rng.uniform(0.8, 1.0)

        # Exponential degradation
        time_steps = np.arange(horizon)
        degradation = initial_rul * np.exp(-decay_rate * time_steps)

        targets[i, :, 0] = degradation
        # Predictions have some lag/noise
        preds[i, :, 0] = degradation + rng.randn(horizon) * 0.05
        preds[i, :, 0] = np.clip(preds[i, :, 0], 0.0, 1.0)

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
    }


@pytest.fixture
def forecasting_data_fault_onset():
    """
    Forecasting data with fault onset: Sudden degradation.

    Shape: (batch=5, time=50, features=1)
    Values: Healthy phase followed by rapid degradation

    **Interesting aspects**:
    - Tests fault detection capability
    - Sudden change in degradation rate
    - Model should track rapid changes
    """
    batch_size = 5
    horizon = 50
    fault_point = 30
    rng = np.random.RandomState(789)

    preds = np.zeros((batch_size, horizon, 1))
    targets = np.zeros((batch_size, horizon, 1))

    for i in range(batch_size):
        # Healthy phase: slow degradation
        healthy_phase = np.linspace(1.0, 0.7, fault_point)

        # Fault onset: rapid degradation
        fault_phase = np.linspace(0.7, 0.0, horizon - fault_point)

        degradation = np.concatenate([healthy_phase, fault_phase])
        targets[i, :, 0] = degradation

        # Predictions lag behind (typical model behavior)
        lag = 3
        preds[i, :lag, 0] = degradation[:lag]
        preds[i, lag:, 0] = degradation[:-lag] + rng.randn(horizon - lag) * 0.1
        preds[i, :, 0] = np.clip(preds[i, :, 0], 0.0, 1.0)

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
    }


@pytest.fixture
def forecasting_data_multivariate():
    """
    Multivariate forecasting data: Multiple sensor channels.

    Shape: (batch=8, time=20, features=5)
    Values: Multiple correlated sensor readings

    **Interesting aspects**:
    - Multiple features (sensors)
    - Correlated channels (realistic PHM)
    - Different scales per sensor
    """
    batch_size = 8
    horizon = 20
    n_features = 5
    rng = np.random.RandomState(321)

    preds = np.zeros((batch_size, horizon, n_features))
    targets = np.zeros((batch_size, horizon, n_features))

    # Create correlated sensor data
    for i in range(batch_size):
        base_trend = np.linspace(1.0, 0.3, horizon)

        for f in range(n_features):
            # Each sensor has different characteristics
            sensor_scale = 0.5 + f * 0.2
            sensor_noise = rng.randn(horizon) * 0.1

            targets[i, :, f] = base_trend * sensor_scale + sensor_noise
            preds[i, :, f] = targets[i, :, f] + rng.randn(horizon) * 0.05

        # Normalize to [0, 1]
        targets[i] = (targets[i] - targets[i].min()) / (
            targets[i].max() - targets[i].min() + 1e-8
        )
        preds[i] = (preds[i] - preds[i].min()) / (
            preds[i].max() - preds[i].min() + 1e-8
        )

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
    }


# =============================================================================
# === CLASSIFICATION DATA FIXTURES ===
# =============================================================================


@pytest.fixture
def classification_data_balanced():
    """
    Balanced classification data: Equal class distribution.

    Predictions: (batch=12, time=3, classes=3) - logits
    Targets: (batch=12, time=3, 1) - class labels

    **Interesting aspects**:
    - Balanced classes (equal distribution)
    - Multiple timesteps
    - Clear predictions (high confidence)
    """
    batch_size = 12
    time_steps = 3
    n_classes = 3
    rng = np.random.RandomState(555)

    # Create balanced class distribution
    targets = np.zeros((batch_size, time_steps, 1), dtype=np.int32)
    for t in range(time_steps):
        # Each timestep has balanced classes
        targets[:, t, 0] = np.tile(np.arange(n_classes), batch_size // n_classes)[
            :batch_size
        ]

    # Create logits with high confidence (clear predictions)
    preds = np.zeros((batch_size, time_steps, n_classes))
    for i in range(batch_size):
        for t in range(time_steps):
            true_class = targets[i, t, 0]
            # High logit for true class, low for others
            preds[i, t, true_class] = 10.0
            preds[i, t, :] += rng.randn(n_classes) * 0.5

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.int32),
    }


@pytest.fixture
def classification_data_imbalanced():
    """
    Imbalanced classification data: One dominant class.

    Predictions: (batch=20, time=2, classes=3) - logits
    Targets: (batch=20, time=2, 1) - class labels

    **Interesting aspects**:
    - Highly imbalanced (class 0: 60%, class 1: 30%, class 2: 10%)
    - Tests metric handling of imbalanced data
    - Some ambiguous predictions (low confidence)
    """
    batch_size = 20
    time_steps = 2
    n_classes = 3
    rng = np.random.RandomState(666)

    # Create imbalanced distribution
    targets = np.zeros((batch_size, time_steps, 1), dtype=np.int32)
    # Class 0: 60%, Class 1: 30%, Class 2: 10%
    class_dist = [0] * 12 + [1] * 6 + [2] * 2

    for t in range(time_steps):
        rng.shuffle(class_dist)
        targets[:, t, 0] = np.array(class_dist[:batch_size])

    # Create logits with varying confidence
    preds = np.zeros((batch_size, time_steps, n_classes))
    for i in range(batch_size):
        for t in range(time_steps):
            true_class = targets[i, t, 0]
            # Moderate confidence (some ambiguity)
            preds[i, t, true_class] = 5.0
            preds[i, t, :] += rng.randn(n_classes) * 2.0

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.int32),
    }


@pytest.fixture
def classification_data_misclassified():
    """
    Classification data with misclassifications: Model errors.

    Predictions: (batch=8, time=4, classes=3) - logits
    Targets: (batch=8, time=4, 1) - class labels

    **Interesting aspects**:
    - 25% misclassification rate
    - Tests error handling and metrics
    - Realistic model mistakes
    """
    batch_size = 8
    time_steps = 4
    n_classes = 3
    rng = np.random.RandomState(777)

    # True labels
    targets = rng.randint(0, n_classes, size=(batch_size, time_steps, 1))

    # Predictions with intentional mistakes
    preds = np.zeros((batch_size, time_steps, n_classes))
    misclass_rate = 0.25

    for i in range(batch_size):
        for t in range(time_steps):
            true_class = targets[i, t, 0]

            if rng.random() < misclass_rate:
                # Misclassification: predict wrong class with high confidence
                wrong_class = (true_class + 1) % n_classes
                preds[i, t, wrong_class] = 10.0
            else:
                # Correct prediction
                preds[i, t, true_class] = 10.0

            preds[i, t, :] += rng.randn(n_classes) * 0.3

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.int32),
    }


# =============================================================================
# === MULTI-UNIT DATA FIXTURES ===
# =============================================================================


@pytest.fixture
def multiunit_data_1d_units():
    """
    Multi-unit data with 1D unit IDs: Simple unit identification.

    Predictions: (batch=15, time=1, features=1)
    Targets: (batch=15, time=1, features=1)
    Unit IDs: (batch=15,) - 1D array

    **Interesting aspects**:
    - Multiple units (units 1, 2, 3, 4, 5)
    - Uneven distribution (some units have more samples)
    - Different degradation rates per unit
    """
    batch_size = 15
    _n_units = 5
    rng = np.random.RandomState(888)

    # Create unit IDs with uneven distribution
    # Unit 1: 5 samples, Unit 2: 4 samples, Unit 3: 3 samples, Unit 4: 2 samples, Unit 5: 1 sample
    unit_ids = np.array([1] * 5 + [2] * 4 + [3] * 3 + [4] * 2 + [5] * 1)
    rng.shuffle(unit_ids)

    preds = np.zeros((batch_size, 1, 1))
    targets = np.zeros((batch_size, 1, 1))

    # Each unit has different baseline RUL
    unit_baselines = {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.1}

    for i, unit_id in enumerate(unit_ids):
        baseline = unit_baselines[unit_id]
        targets[i, 0, 0] = baseline + rng.randn() * 0.05
        preds[i, 0, 0] = targets[i, 0, 0] + rng.randn() * 0.1
        targets[i, 0, 0] = np.clip(targets[i, 0, 0], 0.0, 1.0)
        preds[i, 0, 0] = np.clip(preds[i, 0, 0], 0.0, 1.0)

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
        "unit_id": unit_ids.astype(np.int32),
    }


@pytest.fixture
def multiunit_data_2d_units():
    """
    Multi-unit data with 2D unit IDs: Hierarchical identification (dataset, unit).

    Predictions: (batch=12, time=1, features=1)
    Targets: (batch=12, time=1, features=1)
    Unit IDs: (batch=12, 2) - 2D array (dataset_id, unit_id)

    **Interesting aspects**:
    - Hierarchical unit identification
    - Multiple datasets (dataset 1 and 2)
    - Same unit IDs across datasets (tests tuple handling)
    """
    batch_size = 12
    rng = np.random.RandomState(999)

    # Create 2D unit IDs: (dataset_id, unit_id)
    # Dataset 1: units 1, 2, 3
    # Dataset 2: units 1, 2, 3 (same unit IDs, different dataset)
    unit_ids = np.array(
        [
            [1, 1],
            [1, 1],
            [1, 2],
            [1, 2],
            [1, 3],  # Dataset 1
            [2, 1],
            [2, 1],
            [2, 2],
            [2, 2],
            [2, 3],
            [2, 3],
            [2, 3],  # Dataset 2
        ]
    )
    rng.shuffle(unit_ids)

    preds = np.zeros((batch_size, 1, 1))
    targets = np.zeros((batch_size, 1, 1))

    # Different baselines per (dataset, unit) combination
    baselines = {
        (1, 1): 0.8,
        (1, 2): 0.6,
        (1, 3): 0.4,
        (2, 1): 0.7,
        (2, 2): 0.5,
        (2, 3): 0.3,
    }

    for i, (dataset_id, unit_id) in enumerate(unit_ids):
        key = (int(dataset_id), int(unit_id))
        baseline = baselines.get(key, 0.5)
        targets[i, 0, 0] = baseline + rng.randn() * 0.05
        preds[i, 0, 0] = targets[i, 0, 0] + rng.randn() * 0.1
        targets[i, 0, 0] = np.clip(targets[i, 0, 0], 0.0, 1.0)
        preds[i, 0, 0] = np.clip(preds[i, 0, 0], 0.0, 1.0)

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
        "unit_id": unit_ids.astype(np.int32),
    }


@pytest.fixture
def multiunit_data_many_units():
    """
    Multi-unit data with many units: Stress test for unit tracking.

    Predictions: (batch=50, time=1, features=1)
    Targets: (batch=50, time=1, features=1)
    Unit IDs: (batch=50,) - 20 different units

    **Interesting aspects**:
    - Many units (20 different units)
    - Sparse samples per unit (1-3 samples each)
    - Tests metric aggregation across many units
    """
    batch_size = 50
    n_units = 20
    rng = np.random.RandomState(1111)

    # Create unit IDs: each unit appears 1-3 times
    unit_ids = []
    for unit_id in range(1, n_units + 1):
        n_samples = rng.randint(1, 4)
        unit_ids.extend([unit_id] * n_samples)

    unit_ids = np.array(unit_ids[:batch_size])
    rng.shuffle(unit_ids)

    preds = np.zeros((batch_size, 1, 1))
    targets = np.zeros((batch_size, 1, 1))

    for i, unit_id in enumerate(unit_ids):
        # Each unit has unique baseline
        baseline = (unit_id % 10) / 10.0
        targets[i, 0, 0] = baseline + rng.randn() * 0.05
        preds[i, 0, 0] = targets[i, 0, 0] + rng.randn() * 0.1
        targets[i, 0, 0] = np.clip(targets[i, 0, 0], 0.0, 1.0)
        preds[i, 0, 0] = np.clip(preds[i, 0, 0], 0.0, 1.0)

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
        "unit_id": unit_ids.astype(np.int32),
    }


@pytest.fixture
def multiunit_data_time_series():
    """
    Multi-unit data with time series: Multiple timesteps per unit.

    Predictions: (batch=20, time=10, features=1)
    Targets: (batch=20, time=10, features=1)
    Unit IDs: (batch=20,) - 4 units with time series

    **Interesting aspects**:
    - Time series data (multiple timesteps)
    - Multiple units with different degradation patterns
    - Tests per-unit metric aggregation over time
    """
    batch_size = 20
    time_steps = 10
    n_units = 4
    rng = np.random.RandomState(2222)

    # Each unit appears multiple times (time series)
    unit_ids = np.repeat(np.arange(1, n_units + 1), batch_size // n_units)
    unit_ids = np.concatenate([unit_ids, [1] * (batch_size - len(unit_ids))])

    preds = np.zeros((batch_size, time_steps, 1))
    targets = np.zeros((batch_size, time_steps, 1))

    for unit_id in range(1, n_units + 1):
        unit_mask = unit_ids == unit_id
        unit_indices = np.where(unit_mask)[0]

        if len(unit_indices) == 0:
            continue

        # Each unit has different degradation pattern
        if unit_id == 1:
            # Linear degradation
            trend = np.linspace(1.0, 0.2, time_steps)
        elif unit_id == 2:
            # Exponential degradation
            trend = np.exp(-np.linspace(0, 2, time_steps))
        elif unit_id == 3:
            # Step function (sudden failure)
            trend = np.concatenate([np.ones(7), np.zeros(3)])
        else:
            # Constant (healthy)
            trend = np.ones(time_steps) * 0.9

        for idx in unit_indices:
            targets[idx, :, 0] = trend + rng.randn(time_steps) * 0.05
            preds[idx, :, 0] = targets[idx, :, 0] + rng.randn(time_steps) * 0.1
            targets[idx, :, 0] = np.clip(targets[idx, :, 0], 0.0, 1.0)
            preds[idx, :, 0] = np.clip(preds[idx, :, 0], 0.0, 1.0)

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
        "unit_id": unit_ids.astype(np.int32),
    }


# =============================================================================
# === EDGE CASE DATA FIXTURES ===
# =============================================================================


@pytest.fixture
def data_empty_batch():
    """
    Empty batch data: Edge case with no samples.

    **Interesting aspects**:
    - Tests graceful handling of empty input
    - Edge case for all evaluators
    """
    return {
        "predictions": np.zeros((0, 1, 1), dtype=np.float32),
        "targets": np.zeros((0, 1, 1), dtype=np.float32),
    }


@pytest.fixture
def data_perfect_predictions():
    """
    Perfect predictions: Zero error case.

    **Interesting aspects**:
    - Tests lower bound of metrics (should be 0)
    - Ideal model performance
    """
    batch_size = 10
    preds = np.linspace(0.1, 0.9, batch_size).reshape(batch_size, 1, 1)
    targets = preds.copy()

    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
    }


@pytest.fixture
def data_extreme_values():
    """
    Extreme value data: Boundary conditions.

    **Interesting aspects**:
    - Values at boundaries (0.0, 1.0)
    - Tests clipping and normalization handling
    """
    return {
        "predictions": np.array(
            [[[0.0]], [[1.0]], [[0.5]], [[0.01]], [[0.99]]], dtype=np.float32
        ),
        "targets": np.array(
            [[[0.0]], [[1.0]], [[0.5]], [[0.02]], [[0.98]]], dtype=np.float32
        ),
    }


# =============================================================================
# === HELPER FUNCTIONS ===
# =============================================================================


def create_model_out(
    predictions: np.ndarray,
    targets: np.ndarray,
    unit_id: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Helper function to create model_out dictionary.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    targets : np.ndarray
        Ground truth targets
    unit_id : Optional[np.ndarray]
        Optional unit IDs for multi-unit evaluators

    Returns
    -------
    Dict[str, Any]
        model_out dictionary ready for evaluator.update()
    """
    model_out = {
        "predictions": predictions,
        "targets": targets,
    }

    if unit_id is not None:
        model_out["unit_id"] = unit_id

    return model_out
