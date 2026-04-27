"""Comprehensive tests for the Linear Model components.

This module provides rigorous testing for LinearModel and Linear_Forecaster classes
which provide simple but effective baseline models for PHM forecasting tasks.

PHM Context:
-----------
Linear models serve as important baselines in PHM:
- **LinearModel**: Autoregressive linear predictor that uses weighted sum of
  recent observations to predict future values
- **Linear_Forecaster**: Wraps LinearModel in the Forecaster framework for
  integration with the training pipeline

Linear baselines are valuable for:
1. Establishing performance floors for complex models
2. Quick sanity checks on data quality
3. Interpretable trend analysis in degradation data

Reference: Zeng et al. (2022) "Are Transformers Effective for Time Series Forecasting?"

Test Coverage Strategy:
----------------------
1. **LinearModel Tests**: Weight initialization, forward pass, autoregressive prediction
2. **Linear_Forecaster Tests**: Integration with Forecaster, loss computation
3. **Edge Cases**: Empty sequences, single timestep, large prediction horizons
"""

import pytest
import torch
from typing import Dict, Any

from picid.model.forecasters.linear_model.linear_ar import LinearModel
from picid.model.forecasters.linear_model.linear_model import Linear_Forecaster
from picid.evaluator.base import AbstractEvaluator


# =============================================================================
# TEST FIXTURES
# =============================================================================


class MockEvaluator(AbstractEvaluator):
    """Mock evaluator for testing purposes."""

    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}

    def reset(self) -> None:
        self.results.clear()

    def compute(self, mode, epoch, step):
        return self.results

    def update(self, model_outs):
        self.results = {"loss": 0.0}
        return self.results


@pytest.fixture
def mock_optimizer_factory():
    """Factory that creates Adam optimizer."""

    def factory(params):
        return torch.optim.Adam(params, lr=0.001)

    return factory


@pytest.fixture
def mock_scheduler_factory():
    """Factory that creates StepLR scheduler."""

    def factory(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    return factory


# =============================================================================
# LINEARMODEL TESTS
# =============================================================================


class TestLinearModel:
    """Tests for LinearModel class.

    LinearModel performs autoregressive linear prediction using
    learned weights over a fixed context window.
    """

    def test_init_basic(self):
        """Test basic initialization of LinearModel.

        **PHM Logic**: LinearModel needs context_points (window size)
        and d_yt (output dimension) to create appropriate weight matrices.

        **Methodology**: Create LinearModel with specific parameters.

        **Expected**: Weights and bias initialized with correct shapes.

        Validates: Requirement L-1.1 - Basic initialization
        """
        model = LinearModel(context_points=10, shared_weights=False, d_yt=3)

        assert model.window == 10
        assert model.d_yt == 3
        assert model.shared_weights is False
        # Weights: (context_points, d_yt) = (10, 3)
        assert model.weights.shape == (10, 3)
        # Bias: (d_yt,) = (3,)
        assert model.bias.shape == (3,)

    def test_init_shared_weights(self):
        """Test initialization with shared weights.

        **PHM Logic**: Shared weights use the same parameters across
        all output dimensions, reducing model complexity.

        **Methodology**: Create LinearModel with shared_weights=True.

        **Expected**: Single weight column for all outputs.

        Validates: Requirement L-1.2 - Shared weights mode
        """
        model = LinearModel(context_points=10, shared_weights=True, d_yt=5)

        assert model.shared_weights is True
        # With shared weights: (context_points, 1)
        assert model.weights.shape == (10, 1)
        assert model.bias.shape == (1,)

    def test_weight_initialization_range(self):
        """Test that weights are initialized within expected range.

        **PHM Logic**: Proper initialization is crucial for stable training.
        Weights should be initialized based on input dimension.

        **Methodology**: Check weight values are in [-d, d] where d = sqrt(1/context_points).

        **Expected**: All weights within expected bounds.

        Validates: Requirement L-1.3 - Weight initialization
        """
        context_points = 100
        model = LinearModel(context_points=context_points, d_yt=2)

        d = (1.0 / context_points) ** 0.5

        assert model.weights.min() >= -d
        assert model.weights.max() <= d
        assert model.bias.min() >= -d
        assert model.bias.max() <= d

    def test_forward_single_step(self):
        """Test forward pass for single-step prediction.

        **PHM Logic**: Single-step prediction uses the entire context
        window to predict one future value.

        **Methodology**: Pass context tensor, predict 1 step ahead.

        **Expected**: Output shape (B, 1, d_yt).

        Validates: Requirement L-1.4 - Single-step prediction
        """
        model = LinearModel(context_points=10, d_yt=2)

        # Batch=2, Length=10, d_yc=2
        y_c = torch.randn(2, 10, 2)

        output = model(y_c, pred_len=1, d_yt=2)

        assert output.shape == (2, 1, 2)

    def test_forward_multi_step(self):
        """Test forward pass for multi-step prediction.

        **PHM Logic**: Multi-step prediction iteratively predicts and
        feeds back predictions for longer horizons. Critical for RUL.

        **Methodology**: Pass context, predict multiple steps.

        **Expected**: Output shape (B, pred_len, d_yt).

        Validates: Requirement L-1.5 - Multi-step prediction
        """
        model = LinearModel(context_points=10, d_yt=2)

        y_c = torch.randn(2, 20, 2)  # Longer than context window

        output = model(y_c, pred_len=5, d_yt=2)

        assert output.shape == (2, 5, 2)

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic.

        **PHM Logic**: Deterministic outputs are essential for
        reproducible predictions in PHM applications.

        **Methodology**: Run forward twice with same input.

        **Expected**: Identical outputs.

        Validates: Requirement L-1.6 - Determinism
        """
        model = LinearModel(context_points=10, d_yt=2)
        model.eval()

        y_c = torch.randn(2, 15, 2)

        with torch.no_grad():
            output1 = model(y_c, pred_len=3, d_yt=2)
            output2 = model(y_c, pred_len=3, d_yt=2)

        torch.testing.assert_close(output1, output2)

    def test_forward_shared_weights(self):
        """Test forward pass with shared weights mode.

        **PHM Logic**: Shared weights apply same transformation
        to all features - useful when features are homogeneous.

        **Methodology**: Create model with shared_weights, run forward.

        **Expected**: Correct output shape with shared computation.

        Validates: Requirement L-1.7 - Shared weights forward
        """
        model = LinearModel(context_points=10, shared_weights=True, d_yt=3)

        y_c = torch.randn(2, 15, 3)

        output = model(y_c, pred_len=4, d_yt=3)

        assert output.shape == (2, 4, 3)

    def test_inner_forward_computation(self):
        """Test _inner_forward with known weights.

        **PHM Logic**: Verify the linear computation: y = Wx + b

        **Methodology**: Set known weights, compute with known input.

        **Expected**: Output matches manual calculation.

        Validates: Requirement L-1.8 - Inner computation accuracy
        """
        model = LinearModel(context_points=3, shared_weights=False, d_yt=1)

        with torch.no_grad():
            model.weights.data = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
            model.bias.data = torch.tensor([0.0])

        # Input shape: (B, T, D) where T >= window
        inp = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)

        # _inner_forward takes last `window` elements
        result = model._inner_forward(inp)

        # Expected: 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        expected = torch.tensor([[14.0]])
        torch.testing.assert_close(result, expected)


# =============================================================================
# LINEAR_FORECASTER TESTS
# =============================================================================


class TestLinearForecaster:
    """Tests for Linear_Forecaster class.

    Linear_Forecaster wraps LinearModel in the Forecaster framework
    for integration with the training pipeline.
    """

    def test_init_basic(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test basic initialization of Linear_Forecaster.

        **PHM Logic**: Linear_Forecaster inherits from Forecaster and
        adds a LinearModel for the actual predictions.

        **Methodology**: Create Linear_Forecaster with valid parameters.

        **Expected**: Model attribute is LinearModel instance.

        Validates: Requirement L-2.1 - Linear_Forecaster initialization
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Linear_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            context_points=10,
            task_type="forecasting",
            loss="mse",
            evaluators=evaluators,
        )

        assert hasattr(forecaster, "model")
        assert isinstance(forecaster.model, LinearModel)
        assert forecaster.model.window == 10

    def test_forward_kwargs_properties(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test train and eval step forward kwargs.

        **PHM Logic**: Linear model doesn't require special kwargs
        for train/eval modes (no dropout, etc.).

        **Methodology**: Check property values.

        **Expected**: Empty dicts for both.

        Validates: Requirement L-2.2 - Forward kwargs
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Linear_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            context_points=10,
            task_type="forecasting",
            evaluators=evaluators,
        )

        assert forecaster.train_step_forward_kwargs == {}
        assert forecaster.eval_step_forward_kwargs == {}

    def test_forward_model_pass(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test forward_model_pass method.

        **PHM Logic**: forward_model_pass uses y_c (context targets)
        to predict y_t (future targets).

        **Methodology**: Call forward_model_pass with tensors.

        **Expected**: Output has same shape as y_t.

        Validates: Requirement L-2.3 - Forward model pass
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Linear_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            context_points=10,
            task_type="forecasting",
            evaluators=evaluators,
        )

        x_c = torch.randn(2, 10, 4)
        y_c = torch.randn(2, 10, 2)
        x_t = torch.randn(2, 5, 4)
        y_t = torch.randn(2, 5, 2)

        (output,) = forecaster.forward_model_pass(x_c, y_c, x_t, y_t)

        assert output.shape == y_t.shape

    def test_full_forward(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test complete forward pass through forecaster.

        **PHM Logic**: Full forward includes preprocessing (revin, decomp)
        and postprocessing (linear residual).

        **Methodology**: Call forward() method.

        **Expected**: Output shape matches y_t shape.

        Validates: Requirement L-2.4 - Full forward pass
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Linear_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            context_points=10,
            task_type="forecasting",
            evaluators=evaluators,
        )

        x_c = torch.randn(2, 10, 4)
        y_c = torch.randn(2, 10, 2)
        x_t = torch.randn(2, 5, 4)
        y_t = torch.randn(2, 5, 2)

        (output,) = forecaster(x_c, y_c, x_t, y_t)

        assert output.shape == y_t.shape

    def test_with_shared_weights(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test Linear_Forecaster with shared weights option.

        **PHM Logic**: Shared weights reduce parameters when all
        output features should behave similarly.

        **Methodology**: Create with linear_shared_weights=True.

        **Expected**: Model uses shared weights.

        Validates: Requirement L-2.5 - Shared weights option
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Linear_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            context_points=10,
            task_type="forecasting",
            linear_shared_weights=True,
            evaluators=evaluators,
        )

        assert forecaster.model.shared_weights is True


# =============================================================================
# EDGE CASES
# =============================================================================


class TestLinearModelEdgeCases:
    """Tests for edge cases in LinearModel."""

    def test_minimal_context(self):
        """Test with minimal context window (1 point).

        **PHM Logic**: Edge case where only one point is used for prediction.

        **Methodology**: Create model with context_points=1.

        **Expected**: Still produces valid output.

        Validates: Requirement L-3.1 - Minimal context
        """
        model = LinearModel(context_points=1, d_yt=2)

        y_c = torch.randn(2, 5, 2)
        output = model(y_c, pred_len=3, d_yt=2)

        assert output.shape == (2, 3, 2)
        assert not torch.isnan(output).any()

    def test_large_prediction_horizon(self):
        """Test with large prediction horizon.

        **PHM Logic**: Long-term predictions are important for
        early warning in PHM systems.

        **Methodology**: Request pred_len much larger than context.

        **Expected**: Model produces requested output length.

        Validates: Requirement L-3.2 - Large horizon
        """
        model = LinearModel(context_points=10, d_yt=2)

        y_c = torch.randn(2, 20, 2)
        output = model(y_c, pred_len=100, d_yt=2)

        assert output.shape == (2, 100, 2)

    def test_batch_size_one(self):
        """Test with single sample batch.

        **PHM Logic**: Single-sample inference is common during
        real-time monitoring.

        **Methodology**: Pass batch with size 1.

        **Expected**: Correct output shape.

        Validates: Requirement L-3.3 - Single sample batch
        """
        model = LinearModel(context_points=10, d_yt=2)

        y_c = torch.randn(1, 15, 2)
        output = model(y_c, pred_len=5, d_yt=2)

        assert output.shape == (1, 5, 2)

    def test_gradient_flow(self):
        """Test that gradients flow through the model.

        **PHM Logic**: Training requires gradients to update weights.

        **Methodology**: Compute loss, check gradients exist.

        **Expected**: Non-None gradients for weights and bias.

        Validates: Requirement L-3.4 - Gradient flow
        """
        model = LinearModel(context_points=10, d_yt=2)

        y_c = torch.randn(2, 15, 2, requires_grad=True)
        output = model(y_c, pred_len=5, d_yt=2)

        loss = output.sum()
        loss.backward()

        assert model.weights.grad is not None
        assert model.bias.grad is not None
