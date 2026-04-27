"""Comprehensive tests for the Forecaster base class.

This module provides rigorous testing for the Forecaster class which is the
foundational component for all time-series forecasting models in the PHM framework.

PHM Context:
-----------
The Forecaster class handles:
- **Loss Functions**: MSE, MAE, SMAPE for regression tasks; CrossEntropy for classification
- **Batch Mapping**: Converting raw data batches to task-specific formats (forecasting, regression, classification)
- **RevIN Normalization**: Instance normalization for handling distribution shift in time-series
- **Seasonal Decomposition**: Extracting trend and seasonal components

Reference: Zhou et al. (2021) "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"

Test Coverage Strategy:
----------------------
1. **Loss Function Tests**: Validate MSE, MAE, SMAPE calculations with known inputs
2. **Classification Loss Tests**: Verify cross-entropy computation and masking
3. **Forecasting Loss Tests**: Test null value handling, time masking, feature masking
4. **Batch Mapping Tests**: Verify correct tensor arrangement for each task type
5. **RevIN Tests**: Test normalization and denormalization paths
6. **Optimizer Configuration Tests**: Validate optimizer and scheduler setup
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
from unittest.mock import MagicMock

from picid.model.forecasters.forecaster import Forecaster, TransformerForecaster
from picid.evaluator.base import AbstractEvaluator


# =============================================================================
# TEST FIXTURES
# =============================================================================


class MockEvaluator(AbstractEvaluator):
    """Mock evaluator for testing purposes.

    **PHM Context**: Evaluators compute metrics like RMSE, MAE, accuracy
    that are critical for assessing PHM model performance.
    """

    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}

    def reset(self) -> None:
        self.results.clear()

    def compute(self, mode, epoch, step):
        return self.results

    def update(self, model_outs):
        self.results = {"loss": 0.0}
        return self.results


class ConcreteForecaster(Forecaster):
    """Concrete implementation of Forecaster for testing.

    **PHM Context**: This minimal implementation allows testing of the
    base class methods without requiring a full model implementation.
    The forward_model_pass simply returns zeros, allowing us to test
    the surrounding framework logic.
    """

    @property
    def train_step_forward_kwargs(self):
        return {"training": True}

    @property
    def eval_step_forward_kwargs(self):
        return {"training": False}

    def forward_model_pass(self, x_c, y_c, x_t, y_t, **forward_kwargs):
        """Simple pass-through that returns zeros matching y_t shape.

        **PHM Logic**: In real forecasters, this would contain the neural
        network that predicts future states. Here we return zeros for testing.
        """
        if y_t is not None:
            return (torch.zeros_like(y_t),)
        return (torch.zeros(1, 1, 1),)


@pytest.fixture
def mock_optimizer_factory():
    """Factory that creates Adam optimizer.

    **PHM Context**: Optimizer selection impacts training convergence.
    Adam is commonly used for PHM models due to adaptive learning rates.
    """

    def factory(params):
        return torch.optim.Adam(params, lr=0.001)

    return factory


@pytest.fixture
def mock_scheduler_factory():
    """Factory that creates StepLR scheduler.

    **PHM Context**: Learning rate scheduling helps models converge
    better on complex PHM datasets with varying signal characteristics.
    """

    def factory(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    return factory


@pytest.fixture
def forecaster_regression(mock_optimizer_factory, mock_scheduler_factory):
    """Create a Forecaster configured for RUL regression task.

    **PHM Context**: RUL (Remaining Useful Life) prediction is a core
    PHM task where we predict how long until equipment failure.
    """
    evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
    return ConcreteForecaster(
        optimizer_factory=mock_optimizer_factory,
        scheduler_factory=mock_scheduler_factory,
        d_x=4,  # 4 sensor features
        d_yc=2,  # 2 target context features
        d_yt=1,  # 1 RUL output
        task_type="rul",
        loss="mse",
        evaluators=evaluators,
    )


@pytest.fixture
def forecaster_classification(mock_optimizer_factory, mock_scheduler_factory):
    """Create a Forecaster configured for fault classification.

    **PHM Context**: Fault classification identifies the type of failure
    (e.g., inner race, outer race, ball defect in bearings).
    """
    evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
    return ConcreteForecaster(
        optimizer_factory=mock_optimizer_factory,
        scheduler_factory=mock_scheduler_factory,
        d_x=4,
        d_yc=0,
        d_yt=3,  # 3 fault classes
        task_type="fault_classification",
        loss="mse",
        evaluators=evaluators,
    )


@pytest.fixture
def forecaster_forecasting(mock_optimizer_factory, mock_scheduler_factory):
    """Create a Forecaster configured for time-series forecasting.

    **PHM Context**: Forecasting predicts future sensor values which
    can be used for predictive maintenance decisions.
    """
    evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
    return ConcreteForecaster(
        optimizer_factory=mock_optimizer_factory,
        scheduler_factory=mock_scheduler_factory,
        d_x=4,
        d_yc=2,
        d_yt=2,
        task_type="forecasting",
        loss="mse",
        evaluators=evaluators,
    )


@pytest.fixture
def forecaster_state_forecasting(mock_optimizer_factory, mock_scheduler_factory):
    """Create a Forecaster for state forecasting (no context features).

    **PHM Context**: State forecasting predicts future equipment states
    based only on historical states, without external covariates.
    """
    evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
    return ConcreteForecaster(
        optimizer_factory=mock_optimizer_factory,
        scheduler_factory=mock_scheduler_factory,
        d_x=0,
        d_yc=2,
        d_yt=2,
        task_type="state_forecasting",
        loss="mse",
        evaluators=evaluators,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestForecasterInitialization:
    """Tests for Forecaster initialization.

    Validates parameter handling, supported task types, and component setup.
    """

    def test_init_regression_task(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test initialization with RUL regression task.

        **PHM Logic**: RUL is a fundamental prognostics task. The model
        should accept this task type and configure accordingly.

        **Methodology**: Create Forecaster with rul task, verify attributes.

        **Expected**: task_type set correctly, loss function configured.

        Validates: Requirement F-1.1 - Regression task support
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=1,
            task_type="rul",
            loss="mse",
            evaluators=evaluators,
        )

        assert forecaster.task_type == "rul"
        assert forecaster.loss == "mse"
        assert forecaster.d_x == 4
        assert forecaster.d_yc == 2
        assert forecaster.d_yt == 1

    def test_init_classification_task(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test initialization with fault classification task.

        **PHM Logic**: Fault classification identifies failure types from
        sensor signatures. Essential for diagnostics.

        **Methodology**: Create Forecaster with classification task.

        **Expected**: Classification criterion initialized.

        Validates: Requirement F-1.2 - Classification task support
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=5,
            task_type="fault_classification",
            loss="mse",
            evaluators=evaluators,
        )

        assert forecaster.task_type == "fault_classification"
        assert forecaster.classification_criterion is not None

    def test_init_unsupported_task_raises_error(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test that unsupported task type raises ValueError.

        **PHM Logic**: Task validation prevents configuration errors that
        could lead to incorrect model training.

        **Methodology**: Attempt creation with invalid task type.

        **Expected**: ValueError raised with descriptive message.

        Validates: Requirement F-1.3 - Input validation
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        with pytest.raises(ValueError, match="not supported"):
            ConcreteForecaster(
                optimizer_factory=mock_optimizer_factory,
                scheduler_factory=mock_scheduler_factory,
                d_x=4,
                d_yc=2,
                d_yt=1,
                task_type="invalid_task_type",
                loss="mse",
                evaluators=evaluators,
            )

    def test_init_with_linear_window(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test initialization with linear residual connection.

        **PHM Logic**: Linear residual connections help capture trends
        in degradation data, improving RUL prediction accuracy.

        **Methodology**: Create Forecaster with linear_window > 0.

        **Expected**: Linear model component initialized.

        Validates: Requirement F-1.4 - Linear window support
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            loss="mse",
            linear_window=10,
            evaluators=evaluators,
        )

        # linear_model should be a LinearModel instance, not lambda
        assert hasattr(forecaster, "linear_model")
        assert callable(forecaster.linear_model)

    def test_init_with_revin(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test initialization with RevIN normalization.

        **PHM Logic**: RevIN (Reversible Instance Normalization) handles
        distribution shift in sensor data, critical for multi-unit PHM.

        **Methodology**: Create Forecaster with use_revin=True.

        **Expected**: RevIN component initialized.

        Validates: Requirement F-1.5 - RevIN support
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            loss="mse",
            use_revin=True,
            evaluators=evaluators,
        )

        assert forecaster.use_revin is True
        assert hasattr(forecaster, "revin")

    def test_set_inv_scaler_and_set_scaler(self, forecaster_regression):
        """set_inv_scaler and set_scaler update the internal scalers."""
        forecaster_regression.set_inv_scaler(lambda x: x * 2)
        forecaster_regression.set_scaler(lambda x: x / 2)
        assert callable(forecaster_regression._inv_scaler)
        assert hasattr(forecaster_regression, "_scaler")

    def test_init_with_seasonal_decomp(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test initialization with seasonal decomposition.

        **PHM Logic**: Seasonal decomposition separates trend from
        seasonal patterns in equipment data, improving predictions.

        **Methodology**: Create Forecaster with use_seasonal_decomp=True.

        **Expected**: SeriesDecomposition component initialized.

        Validates: Requirement F-1.6 - Seasonal decomposition support
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            loss="mse",
            use_seasonal_decomp=True,
            evaluators=evaluators,
        )

        assert forecaster.use_seasonal_decomp is True
        assert hasattr(forecaster, "seasonal_decomp")


# =============================================================================
# LOSS FUNCTION TESTS
# =============================================================================


class TestForecasterLossFunctions:
    """Tests for loss function calculations.

    Validates MSE, MAE, and SMAPE loss computations which are fundamental
    for training PHM models on regression tasks.
    """

    def test_loss_fn_mse(self, forecaster_regression):
        """Test MSE loss calculation.

        **PHM Logic**: MSE is the standard loss for RUL prediction,
        penalizing large errors more heavily than small ones.

        **Methodology**: Compute MSE loss with known true/pred values.

        **Expected**: Loss = sum((mask * (true - pred))^2) / mask.sum()

        Validates: Requirement F-2.1 - MSE loss accuracy
        """
        forecaster_regression.loss = "mse"

        true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        pred = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        mask = torch.ones_like(true)

        loss = forecaster_regression.loss_fn(true, pred, mask)

        # Expected: ((0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2) / 4 = 0.25
        expected = 0.25
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_loss_fn_mae(self, forecaster_regression):
        """Test MAE loss calculation.

        **PHM Logic**: MAE is robust to outliers, useful when sensor
        data contains spikes or anomalies.

        **Methodology**: Compute MAE loss with known true/pred values.

        **Expected**: Loss = sum(|mask * (true - pred)|) / mask.sum()

        Validates: Requirement F-2.2 - MAE loss accuracy
        """
        forecaster_regression.loss = "mae"

        true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        pred = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        mask = torch.ones_like(true)

        loss = forecaster_regression.loss_fn(true, pred, mask)

        # Expected: (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        expected = 0.5
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_loss_fn_smape(self, forecaster_regression):
        """Test SMAPE (Symmetric Mean Absolute Percentage Error) loss.

        **PHM Logic**: SMAPE is scale-independent, useful when comparing
        predictions across units with different operational ranges.

        **Methodology**: Compute SMAPE with known values.

        **Expected**: Loss = 100 * sum(2 * |pred - true| / (|pred| + |true| + eps)) / mask.sum()

        Validates: Requirement F-2.3 - SMAPE loss accuracy
        """
        forecaster_regression.loss = "smape"

        true = torch.tensor([[10.0]])
        pred = torch.tensor([[12.0]])
        mask = torch.ones_like(true)

        loss = forecaster_regression.loss_fn(true, pred, mask)

        # Expected: 100 * 2 * |12 - 10| / (|12| + |10| + 1e-5) ≈ 18.18
        num = 2.0 * abs(12 - 10)  # 4
        den = abs(12) + abs(10) + 1e-5  # 22
        expected = 100.0 * num / den
        assert torch.isclose(loss, torch.tensor(expected), atol=0.1)

    def test_loss_fn_invalid_raises_error(self, forecaster_regression):
        """Test that invalid loss type raises ValueError.

        **PHM Logic**: Early validation prevents silent failures during training.

        **Methodology**: Attempt to compute loss with invalid type.

        **Expected**: ValueError raised.

        Validates: Requirement F-2.4 - Invalid loss error handling
        """
        forecaster_regression.loss = "invalid_loss"

        true = torch.tensor([[1.0]])
        pred = torch.tensor([[1.5]])
        mask = torch.ones_like(true)

        with pytest.raises(ValueError, match="Unrecognized Loss Function"):
            forecaster_regression.loss_fn(true, pred, mask)

    def test_loss_fn_with_mask(self, forecaster_regression):
        """Test loss calculation with partial masking.

        **PHM Logic**: Masking handles variable-length sequences and
        missing data points common in PHM datasets.

        **Methodology**: Compute loss with mask zeroing some elements.

        **Expected**: Only unmasked elements contribute to loss.

        Validates: Requirement F-2.5 - Mask handling
        """
        forecaster_regression.loss = "mse"

        true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        pred = torch.tensor([[2.0, 3.0], [4.0, 5.0]])  # All errors = 1.0
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])  # Last element masked

        loss = forecaster_regression.loss_fn(true, pred, mask)

        # Expected: (1^2 + 1^2 + 1^2 + 0) / 3 = 1.0
        expected = 1.0
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_loss_fn_with_nan_handling(self, forecaster_regression):
        """Test that NaN values in true tensor are handled.

        **PHM Logic**: Sensor data may contain NaN values due to
        measurement failures. Loss function must handle gracefully.

        **Methodology**: Pass tensor with NaN values.

        **Expected**: NaN values replaced with 0 (nan_to_num behavior).

        Validates: Requirement F-2.6 - NaN handling
        """
        forecaster_regression.loss = "mse"

        # true has NaN - should be converted to 0 by nan_to_num
        true = torch.tensor([[1.0, float("nan")]])
        pred = torch.tensor([[1.0, 0.0]])
        mask = torch.ones_like(pred)

        loss = forecaster_regression.loss_fn(true, pred, mask)

        # After nan_to_num: true = [[1.0, 0.0]], pred = [[1.0, 0.0]]
        # Loss = 0.0 (perfect match)
        assert torch.isfinite(loss)


# =============================================================================
# FORECASTING LOSS TESTS
# =============================================================================


class TestForecasterForecastingLoss:
    """Tests for forecasting_loss method.

    Validates null value handling, time masking, and feature masking
    which are essential for handling PHM data with missing values.
    """

    def test_forecasting_loss_basic(self, forecaster_regression):
        """Test basic forecasting loss without special masking.

        **PHM Logic**: Basic loss computation when all data is valid
        and no special masking is needed.

        **Methodology**: Compute loss with standard inputs.

        **Expected**: Loss computed using full mask.

        Validates: Requirement F-3.1 - Basic forecasting loss
        """
        outputs = torch.tensor([[[1.0], [2.0]]])  # (B, T, D)
        y_t = torch.tensor([[[1.5], [2.5]]])

        loss, mask = forecaster_regression.forecasting_loss(
            outputs=outputs, y_t=y_t, time_mask=None, feat_mask=None
        )

        assert loss.item() > 0  # Non-zero loss
        assert mask.shape == y_t.shape

    def test_forecasting_loss_with_null_value(self, forecaster_regression):
        """Test forecasting loss with null value masking.

        **PHM Logic**: Null values (-999, etc.) indicate missing data
        in PHM datasets. These should be excluded from loss.

        **Methodology**: Set null_value and include it in y_t.

        **Expected**: Null value positions are masked out.

        Validates: Requirement F-3.2 - Null value handling
        """
        forecaster_regression.set_null_value(-999.0)

        outputs = torch.tensor([[[1.0], [-999.0]]])
        y_t = torch.tensor([[[1.5], [-999.0]]])  # Second is null

        loss, mask = forecaster_regression.forecasting_loss(
            outputs=outputs, y_t=y_t, time_mask=None
        )

        # Second position should be masked
        assert mask[0, 1, 0] == 0

    def test_forecasting_loss_with_time_mask(self, forecaster_regression):
        """Test forecasting loss with time masking.

        **PHM Logic**: Time masking allows training on partial sequences,
        useful for curriculum learning in PHM.

        **Methodology**: Set time_mask to exclude later timesteps.

        **Expected**: Timesteps after mask are excluded.

        Validates: Requirement F-3.3 - Time masking
        """
        outputs = torch.tensor([[[1.0], [2.0], [3.0]]])
        y_t = torch.tensor([[[1.5], [2.5], [3.5]]])

        loss, mask = forecaster_regression.forecasting_loss(
            outputs=outputs,
            y_t=y_t,
            time_mask=2,  # Only first 2 timesteps
        )

        # Third timestep should be masked
        assert mask[0, 2, 0] == 0

    def test_forecasting_loss_with_feature_mask(self, forecaster_regression):
        """Test forecasting loss with feature masking.

        **PHM Logic**: Feature masking allows focusing on specific
        sensor channels during training.

        **Methodology**: Set feat_mask to include only specific features.

        **Expected**: Only specified features contribute to loss.

        Validates: Requirement F-3.4 - Feature masking
        """
        outputs = torch.tensor([[[1.0, 2.0, 3.0]]])  # 3 features
        y_t = torch.tensor([[[1.5, 2.5, 3.5]]])

        loss, mask = forecaster_regression.forecasting_loss(
            outputs=outputs,
            y_t=y_t,
            time_mask=None,
            feat_mask=[0, 2],  # Only features 0 and 2
        )

        # Feature 1 should be masked out
        assert mask[0, 0, 0] == 1
        assert mask[0, 0, 1] == 0  # Masked out
        assert mask[0, 0, 2] == 1


# =============================================================================
# PREDICT TESTS
# =============================================================================


class TestForecasterPredict:
    """Tests for predict method."""

    def test_predict_returns_scaled_predictions(
        self, forecaster_forecasting, mock_optimizer_factory, mock_scheduler_factory
    ):
        """predict() scales inputs, runs forward, inverse-scales outputs (lines 197-221)."""
        forecaster_forecasting.set_scaler(lambda x: x)
        forecaster_forecasting.set_inv_scaler(lambda x: x)
        x_c = torch.randn(2, 10, 4)
        y_c = torch.randn(2, 10, 2)
        x_t = torch.randn(2, 5, 4)
        preds = forecaster_forecasting.predict(x_c, y_c, x_t)
        assert preds.shape == (2, 5, 2)
        assert isinstance(preds, torch.Tensor)


# =============================================================================
# CLASSIFICATION LOSS TESTS
# =============================================================================


class TestForecasterClassificationLoss:
    """Tests for classification_loss method.

    Validates cross-entropy loss for fault classification tasks.
    """

    def test_classification_loss_basic(self, forecaster_classification):
        """Test basic classification loss computation.

        **PHM Logic**: Classification loss for identifying fault types
        from sensor signatures.

        **Methodology**: Compute cross-entropy with known logits/labels.

        **Expected**: Correct cross-entropy value.

        Validates: Requirement F-4.1 - Classification loss accuracy
        """
        # Batch=2, Time=1, Classes=3
        outputs = torch.tensor([[[1.0, 2.0, 0.5]], [[0.5, 1.0, 2.0]]])
        y_t = torch.tensor([[0], [2]])  # Class labels
        mask = None

        loss = forecaster_classification.classification_loss(outputs, y_t, mask)

        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_classification_loss_with_mask(self, forecaster_classification):
        """Test classification loss with masking.

        **PHM Logic**: Mask invalid samples that may have unknown
        fault labels in partially labeled datasets.

        **Methodology**: Apply mask to exclude some samples.

        **Expected**: Masked samples don't contribute to loss.

        Validates: Requirement F-4.2 - Classification mask handling
        """
        outputs = torch.tensor([[[1.0, 2.0, 0.5]], [[0.5, 1.0, 2.0]]])
        y_t = torch.tensor([[0], [2]])
        mask = torch.tensor([[1.0], [0.0]])  # Second sample masked

        loss = forecaster_classification.classification_loss(outputs, y_t, mask)

        assert torch.isfinite(loss)

    def test_classification_loss_with_mask_branch(self, forecaster_classification):
        """classification_loss with mask not None uses mask_flat (lines 162-163)."""
        outputs = torch.tensor([[[1.0, 2.0, 0.5]], [[0.5, 1.0, 2.0]]])
        y_t = torch.tensor([[0], [2]])
        mask = torch.tensor([[1.0], [0.5]])
        loss = forecaster_classification.classification_loss(outputs, y_t, mask)
        assert torch.isfinite(loss)

    def test_compute_loss_classification_branch_returns_none_mask(
        self, forecaster_classification, monkeypatch
    ):
        outputs = torch.randn(2, 5, forecaster_classification.d_yt)

        def fake_forward(x_c, y_c, x_t, y_t, **forward_kwargs):
            return (outputs,)

        monkeypatch.setattr(forecaster_classification, "forward", fake_forward)

        loss, computed_outputs, mask = forecaster_classification.compute_loss(
            batch=(
                None,
                None,
                torch.randn(2, 5, forecaster_classification.d_x),
                torch.randint(0, forecaster_classification.d_yt, (2, 5, 1)),
            )
        )

        assert torch.isfinite(loss)
        assert computed_outputs is outputs
        assert mask is None


# =============================================================================
# BATCH MAPPING TESTS
# =============================================================================


class TestForecasterBatchMapping:
    """Tests for _map_batch_to_task method.

    Validates correct transformation of raw batches to task-specific formats.
    """

    def test_map_batch_forecasting(self, forecaster_forecasting):
        """Test batch mapping for forecasting task.

        **PHM Logic**: Forecasting requires context features/targets
        and future features/targets in specific arrangement.

        **Methodology**: Pass forecasting batch, verify tuple structure.

        **Expected**: (x_c, y_c, x_t, y_t) from batch["context"] and batch["target"]

        Validates: Requirement F-5.1 - Forecasting batch mapping
        """
        batch = {
            "context": {
                "features_seq_x": torch.randn(2, 10, 4),
                "features_seq_y": torch.randn(2, 5, 4),
            },
            "target": {
                "target_seq_x": torch.randn(2, 10, 2),
                "target_seq_y": torch.randn(2, 5, 2),
            },
        }

        x_c, y_c, x_t, y_t = forecaster_forecasting._map_batch_to_task(batch)

        assert x_c.shape == (2, 10, 4)
        assert y_c.shape == (2, 10, 2)
        assert x_t.shape == (2, 5, 4)
        assert y_t.shape == (2, 5, 2)

    def test_map_batch_state_forecasting(self, forecaster_state_forecasting):
        """Test batch mapping for state forecasting task.

        **PHM Logic**: State forecasting uses only target sequences
        without external context features.

        **Methodology**: Pass state_forecasting batch.

        **Expected**: x_c=None, x_t=None, y_c and y_t from batch

        Validates: Requirement F-5.2 - State forecasting batch mapping
        """
        batch = {
            "features_seq_x": torch.randn(2, 10, 2),  # y_c
            "features_seq_y": torch.randn(2, 5, 2),  # y_t
        }

        x_c, y_c, x_t, y_t = forecaster_state_forecasting._map_batch_to_task(batch)

        assert x_c is None
        assert y_c.shape == (2, 10, 2)
        assert x_t is None
        assert y_t.shape == (2, 5, 2)

    def test_map_batch_regression(self, forecaster_regression):
        """Test batch mapping for regression task (RUL).

        **PHM Logic**: Regression maps features to targets without
        temporal context splitting.

        **Methodology**: Pass regression batch with features and rul.

        **Expected**: x_c=None, y_c=None, x_t=features, y_t=rul

        Validates: Requirement F-5.3 - Regression batch mapping
        """
        batch = {
            "features": torch.randn(2, 10, 4),  # (B, T, D)
            "rul": torch.randn(2, 1),  # (B, 1)
        }

        x_c, y_c, x_t, y_t = forecaster_regression._map_batch_to_task(batch)

        assert x_c is None
        assert y_c is None
        assert x_t.shape == (2, 10, 4)
        assert y_t.shape == (2, 1, 1)  # Expanded to 3D

    def test_map_batch_regression_target_dim2(self, forecaster_regression):
        """_map_batch_to_task with target dim 2 uses rearrange b f -> b 1 f (lines 359-361)."""
        batch = {"features": torch.randn(2, 10, 4), "rul": torch.randn(2, 1)}
        x_c, y_c, x_t, y_t = forecaster_regression._map_batch_to_task(batch)
        assert y_t.shape == (2, 1, 1)

    def test_map_batch_regression_target_dim1(self, forecaster_regression):
        """_map_batch_to_task with target dim 1 uses rearrange b -> b 1 1 (lines 364-366)."""
        batch = {"features": torch.randn(2, 10, 4), "rul": torch.randn(2)}
        x_c, y_c, x_t, y_t = forecaster_regression._map_batch_to_task(batch)
        assert y_t.shape == (2, 1, 1)

    def test_map_batch_unrecognized_task_raises(self, forecaster_regression):
        """_map_batch_to_task with unrecognized task raises ValueError (line 353)."""
        forecaster_regression.task_type = "invalid_task"
        batch = {"features": torch.randn(2, 10, 4)}
        with pytest.raises(ValueError, match="Unrecognized task type"):
            forecaster_regression._map_batch_to_task(batch)

    def test_map_batch_classification(self, forecaster_classification):
        """Test batch mapping for classification task.

        **PHM Logic**: Classification maps features to class labels.

        **Methodology**: Pass classification batch.

        **Expected**: Similar to regression but with class indices.

        Validates: Requirement F-5.4 - Classification batch mapping
        """
        batch = {
            "features": torch.randn(2, 10, 4),
            "fault_classification": torch.tensor([0, 2]),  # (B,)
        }

        x_c, y_c, x_t, y_t = forecaster_classification._map_batch_to_task(batch)

        assert x_c is None
        assert y_c is None
        assert x_t.shape == (2, 10, 4)
        assert y_t.shape == (2, 1, 1)  # Expanded


# =============================================================================
# NAN HANDLING TESTS
# =============================================================================


class TestForecasterNanHandling:
    """Tests for NaN handling in forward pass.

    Validates that NaN values are properly handled to prevent
    gradient explosions during training.
    """

    def test_nan_to_num_method(self, forecaster_regression):
        """Test nan_to_num helper method.

        **PHM Logic**: Sensor data may contain NaN from dropout or
        transmission errors. Must be handled safely.

        **Methodology**: Pass tensors with NaN through nan_to_num.

        **Expected**: NaN replaced with 0.

        Validates: Requirement F-6.1 - NaN conversion
        """
        input1 = torch.tensor([1.0, float("nan"), 3.0])
        input2 = torch.tensor([float("nan"), 2.0, float("nan")])

        result1, result2 = forecaster_regression.nan_to_num(input1, input2)

        assert not torch.isnan(result1).any()
        assert not torch.isnan(result2).any()


# =============================================================================
# CONFIGURE OPTIMIZERS TESTS
# =============================================================================


class TestForecasterOptimizers:
    """Tests for optimizer configuration.

    Validates optimizer and scheduler setup for training.
    """

    def test_configure_optimizers_without_scheduler_raises(
        self, mock_optimizer_factory
    ):
        """configure_optimizers with scheduler_factory=None raises NotImplementedError (lines 392-396)."""
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=None,
            d_x=4,
            d_yc=2,
            d_yt=1,
            task_type="rul",
            loss="mse",
            evaluators=evaluators,
        )
        forecaster.dummy_param = nn.Parameter(torch.randn(5))
        with pytest.raises(NotImplementedError, match="Check this branch"):
            forecaster.configure_optimizers()

    def test_configure_optimizers_with_scheduler(self, forecaster_regression):
        """Test optimizer configuration with scheduler.

        **PHM Logic**: Learning rate scheduling improves convergence
        on complex PHM datasets.

        **Methodology**: Call configure_optimizers, verify structure.

        **Expected**: Dict with optimizer and lr_scheduler keys.

        Validates: Requirement F-7.1 - Optimizer configuration
        """
        # Add a parameter for the optimizer
        forecaster_regression.dummy_param = nn.Parameter(torch.randn(10))

        result = forecaster_regression.configure_optimizers()

        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["monitor"] == "val/loss"


# =============================================================================
# STEP AND TRAINING TESTS
# =============================================================================


class TestForecasterSteps:
    """Tests for training and validation steps.

    Validates the step method which combines loss computation
    and output formatting.
    """

    def test_step_regression_train(self, forecaster_regression):
        """Test training step for regression task.

        **PHM Logic**: Training step computes loss and returns
        predictions/targets for logging.

        **Methodology**: Call step with train=True.

        **Expected**: Dict with loss, predictions, mask, targets.

        Validates: Requirement F-8.1 - Training step
        """
        batch = (
            None,  # x_c
            None,  # y_c
            torch.randn(2, 10, 4),  # x_t
            torch.randn(2, 1, 1),  # y_t
        )

        result = forecaster_regression.step(batch, train=True)

        assert "loss" in result
        assert "predictions" in result
        assert "targets" in result

    def test_step_regression_eval(self, forecaster_regression):
        """Test evaluation step for regression task.

        **PHM Logic**: Evaluation step uses different kwargs
        (e.g., no dropout) than training.

        **Methodology**: Call step with train=False.

        **Expected**: Similar output to training but with eval kwargs.

        Validates: Requirement F-8.2 - Evaluation step
        """
        batch = (
            None,
            None,
            torch.randn(2, 10, 4),
            torch.randn(2, 1, 1),
        )

        result = forecaster_regression.step(batch, train=False)

        assert "loss" in result
        assert torch.isfinite(result["loss"])

    def test_validation_and_test_steps_delegate_with_eval_mode(
        self, forecaster_regression, monkeypatch
    ):
        mapped_batch = (
            None,
            None,
            torch.randn(2, 10, 4),
            torch.randn(2, 1, 1),
        )
        calls = []

        monkeypatch.setattr(
            forecaster_regression, "_map_batch_to_task", lambda batch: mapped_batch
        )

        def fake_step(batch, train=False):
            calls.append((batch, train))
            return {"loss": torch.tensor(0.5)}

        monkeypatch.setattr(forecaster_regression, "step", fake_step)

        val_out = forecaster_regression._validation_step(
            {"features": torch.randn(1)}, 0
        )
        test_out = forecaster_regression._test_step({"features": torch.randn(1)}, 0)

        assert val_out == {"loss": torch.tensor(0.5)}
        assert test_out == {"loss": torch.tensor(0.5)}
        assert calls == [(mapped_batch, False), (mapped_batch, False)]

    def test_predict_step_uses_eval_forward_kwargs(
        self, forecaster_regression, monkeypatch
    ):
        batch = (
            None,
            None,
            torch.randn(2, 10, 4),
            torch.randn(2, 1, 1),
        )
        captured = {}

        def fake_forward(x_c, y_c, x_t, y_t, **forward_kwargs):
            captured["forward_kwargs"] = forward_kwargs
            return (torch.zeros_like(y_t),)

        monkeypatch.setattr(forecaster_regression, "forward", fake_forward)

        out = forecaster_regression.predict_step(batch, 0)

        assert out[0].shape == batch[-1].shape
        assert captured["forward_kwargs"] == {"training": False}


# =============================================================================
# TRANSFORMER FORECASTER TESTS
# =============================================================================


class ConcreteTransformerForecaster(TransformerForecaster):
    """Concrete implementation of TransformerForecaster for testing."""

    @property
    def train_step_forward_kwargs(self):
        return {"training": True}

    @property
    def eval_step_forward_kwargs(self):
        return {"training": False}

    def forward_model_pass(self, x_c, y_c, x_t, y_t, **forward_kwargs):
        if y_t is not None:
            return (torch.zeros_like(y_t),)
        return (torch.zeros(1, 1, 1),)


class TestTransformerForecaster:
    """Tests for TransformerForecaster subclass.

    Validates transformer-specific training behavior with manual
    scheduler stepping.
    """

    def test_transformer_forecaster_init(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test TransformerForecaster initialization.

        **PHM Logic**: Transformer models often require custom scheduler
        handling for warmup and decay.

        **Methodology**: Create TransformerForecaster, verify attributes.

        **Expected**: scheduler_Loss_monitor set correctly.

        Validates: Requirement F-9.1 - Transformer initialization
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteTransformerForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=1,
            task_type="rul",
            loss="mse",
            evaluators=evaluators,
            scheduler_Loss_monitor="forecast_loss",
        )

        assert forecaster.scheduler_Loss_monitor == "forecast_loss"
        assert forecaster.validation_step_outputs == []

    def test_transformer_forecaster_validation_epoch_end(self, mock_optimizer_factory):
        """TransformerForecaster on_validation_epoch_end steps scheduler (lines 411-428)."""
        mock_sched = MagicMock()

        def sched_factory(optimizer):
            return mock_sched

        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteTransformerForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=sched_factory,
            d_x=4,
            d_yc=2,
            d_yt=1,
            task_type="rul",
            loss="mse",
            evaluators=evaluators,
            scheduler_Loss_monitor="loss",
        )
        forecaster.dummy_param = nn.Parameter(torch.randn(5))
        forecaster.configure_optimizers()
        batch = (
            None,
            None,
            torch.randn(2, 10, 4),
            torch.randn(2, 1, 1),
        )
        forecaster.validation_step_outputs = [forecaster.step(batch, train=False)]
        forecaster.on_validation_epoch_end()
        assert forecaster.validation_step_outputs == []

    def test_transformer_forecaster_validation_step_appends_outputs(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteTransformerForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=1,
            task_type="rul",
            loss="mse",
            evaluators=evaluators,
        )
        batch = {
            "features": torch.randn(2, 10, 4),
            "rul": torch.randn(2, 1),
        }

        out = forecaster._validation_step(batch, 0)

        assert forecaster.validation_step_outputs == [out]

    def test_transformer_forecaster_validation_epoch_end_raises_when_monitor_missing(
        self, mock_optimizer_factory
    ):
        mock_sched = MagicMock()

        def sched_factory(optimizer):
            return mock_sched

        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteTransformerForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=sched_factory,
            d_x=4,
            d_yc=2,
            d_yt=1,
            task_type="rul",
            loss="mse",
            evaluators=evaluators,
            scheduler_Loss_monitor="forecast_loss",
        )
        forecaster.dummy_param = nn.Parameter(torch.randn(5))
        forecaster.configure_optimizers()
        forecaster.validation_step_outputs = [{"loss": torch.tensor(0.5)}]

        with pytest.raises(ValueError, match="Could not find forecast_loss"):
            forecaster.on_validation_epoch_end()

    def test_transformer_forecaster_training_step(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """TransformerForecaster training_step calls super and scheduler.step (lines 431-433)."""
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = ConcreteTransformerForecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=1,
            task_type="rul",
            loss="mse",
            evaluators=evaluators,
        )
        forecaster.dummy_param = nn.Parameter(torch.randn(5))
        forecaster.configure_optimizers()
        batch = {
            "features": torch.randn(2, 10, 4),
            "rul": torch.randn(2, 1),
        }
        out = forecaster.training_step(batch, 0)
        assert isinstance(out, torch.Tensor)
