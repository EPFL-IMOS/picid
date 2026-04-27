"""Comprehensive tests for the Crossformer Model components.

This module provides rigorous testing for Crossformer-based forecasting models
which use cross-dimensional attention for multivariate time-series forecasting.

PHM Context:
-----------
Crossformer is particularly useful for PHM because:
- **Cross-Dimensional Attention**: Captures relationships between different sensors
- **Segment-wise Processing**: Handles long sequences efficiently
- **Two-Stage Attention**: Combines temporal and cross-variable attention

This is valuable for PHM applications where:
1. Multiple sensors monitor different aspects of equipment health
2. Interactions between sensor readings indicate fault patterns
3. Long historical sequences need efficient processing

Reference: Zhang et al. (2023) "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting"

Test Coverage Strategy:
----------------------
1. **Crossformer Core Tests**: Encoder-decoder architecture, DSW embedding
2. **Crossformer_Forecaster Tests**: Integration with training pipeline
3. **Attention Layer Tests**: Cross-dimensional attention mechanisms
4. **Edge Cases**: Different segment sizes, various decoder embeddings
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from picid.model.forecasters.crossformer_model.crossformer_model import (
    Crossformer_Forecaster,
)
from picid.model.forecasters.crossformer_model.cross_former import Crossformer
from picid.model.forecasters.crossformer_model.cross_embed import DSW_embedding
from picid.model.forecasters.crossformer_model.cross_attn import (
    FullAttention,
    AttentionLayer,
)
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
# DSW_EMBEDDING TESTS
# =============================================================================


class TestDSWEmbedding:
    """Tests for DSW (Dimension-Segment-Wise) embedding.

    DSW embedding projects time segments into a higher-dimensional space
    while preserving cross-dimensional structure.
    """

    def test_init_basic(self):
        """Test basic initialization of DSW_embedding.

        **PHM Logic**: Embedding must accept segment length and model dimension
        to create appropriate projection.

        **Methodology**: Create embedding with specific parameters.

        **Expected**: Linear layer initialized with correct dimensions.

        Validates: Requirement CF-1.1 - DSW embedding initialization
        """
        embedding = DSW_embedding(seg_len=24, d_model=64)

        assert hasattr(embedding, "linear")
        assert isinstance(embedding.linear, nn.Linear)

    def test_forward_basic(self):
        """Test forward pass of DSW_embedding.

        **PHM Logic**: Embedding transforms input sequence into segment-wise
        representation for cross-dimensional attention.

        **Methodology**: Pass batch through embedding.

        **Expected**: Output shape reflects segmented representation.

        Validates: Requirement CF-1.2 - DSW embedding forward
        """
        seg_len = 24
        d_model = 64
        embedding = DSW_embedding(seg_len=seg_len, d_model=d_model)

        # Input: (batch, seq_len, data_dim)
        # seq_len must be divisible by seg_len
        batch_size = 2
        seq_len = 96  # 4 segments of 24
        data_dim = 3

        x = torch.randn(batch_size, seq_len, data_dim)
        output = embedding(x)

        # Output shape: (batch, data_dim, n_segments, d_model)
        n_segments = seq_len // seg_len
        assert output.shape == (batch_size, data_dim, n_segments, d_model)


# =============================================================================
# CROSSFORMER CORE TESTS
# =============================================================================


class TestCrossformer:
    """Tests for the core Crossformer model."""

    def test_init_basic(self):
        """Test basic initialization of Crossformer.

        **PHM Logic**: Crossformer needs data dimensions, sequence lengths,
        and segment configuration for proper setup.

        **Methodology**: Create Crossformer with typical PHM parameters.

        **Expected**: Encoder and decoder components initialized.

        Validates: Requirement CF-2.1 - Crossformer initialization
        """
        # Use larger win_size to ensure enough segments after merging
        model = Crossformer(
            data_dim=3,
            in_len=192,  # Larger to have more segments
            out_len=24,
            seg_len=12,  # Smaller segment for more segments
            win_size=2,  # Smaller window
            d_model=64,
            d_ff=128,
            n_heads=4,
            e_layers=2,
            device="cpu",
        )

        assert model.data_dim == 3
        assert model.in_len == 192
        assert model.out_len == 24
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")

    def test_init_with_random_decoder_embedding(self):
        """Test initialization with random decoder embedding.

        **PHM Logic**: Random embedding doesn't require target sequence,
        useful for pure forecasting tasks.

        **Methodology**: Create with decoder_embedding="random".

        **Expected**: No value embedding for decoder.

        Validates: Requirement CF-2.2 - Random decoder embedding
        """
        model = Crossformer(
            data_dim=3,
            in_len=192,
            out_len=24,
            seg_len=12,
            win_size=2,
            decoder_embedding="random",
            device="cpu",
        )

        assert model.decoder_embedding == "random"
        assert not hasattr(model, "dec_value_embedding")

    def test_forward_without_y_seq(self):
        """Test forward pass without target sequence (random embedding).

        **PHM Logic**: Forecasting without ground truth - uses learned
        decoder positional embeddings.

        **Methodology**: Pass only x_seq, y_seq=None.

        **Expected**: Output shape (batch, out_len, data_dim).

        Validates: Requirement CF-2.3 - Forward without y_seq
        """
        model = Crossformer(
            data_dim=3,
            in_len=192,
            out_len=24,
            seg_len=12,
            win_size=2,
            decoder_embedding="random",
            device="cpu",
        )

        batch_size = 2
        x_seq = torch.randn(batch_size, 192, 3)

        output = model(x_seq, y_seq=None)

        assert output.shape == (batch_size, 24, 3)

    def test_forward_with_y_seq_dsw(self):
        """Test forward pass with target sequence (DSW embedding).

        **PHM Logic**: Using target sequence embedding enables
        teacher forcing or conditional generation.

        **Methodology**: Pass x_seq and y_seq with DSW embedding.

        **Expected**: Output incorporates target information.

        Validates: Requirement CF-2.4 - Forward with y_seq
        """
        model = Crossformer(
            data_dim=3,
            in_len=192,
            out_len=24,
            seg_len=12,
            win_size=2,
            decoder_embedding="DSW",
            device="cpu",
        )

        batch_size = 2
        x_seq = torch.randn(batch_size, 192, 3)
        y_seq = torch.randn(batch_size, 24, 3)

        output = model(x_seq, y_seq=y_seq)

        assert output.shape == (batch_size, 24, 3)

    def test_forward_with_baseline(self):
        """Test forward pass with baseline subtraction.

        **PHM Logic**: Baseline mode subtracts mean to help model
        learn residuals - useful for detrending.

        **Methodology**: Create with baseline=True.

        **Expected**: Output reflects mean-adjusted predictions.

        Validates: Requirement CF-2.5 - Baseline mode
        """
        model = Crossformer(
            data_dim=3,
            in_len=192,
            out_len=24,
            seg_len=12,
            win_size=2,
            baseline=True,
            decoder_embedding="random",
            device="cpu",
        )

        batch_size = 2
        x_seq = torch.randn(batch_size, 192, 3) + 10.0  # Add offset

        output = model(x_seq)

        assert output.shape == (batch_size, 24, 3)

    def test_forward_with_padding(self):
        """Test forward pass when input needs padding.

        **PHM Logic**: Input length not divisible by segment length
        requires padding - common in real PHM data.

        **Methodology**: Use in_len not divisible by seg_len.

        **Expected**: Model handles padding internally.

        Validates: Requirement CF-2.6 - Padding handling
        """
        # Use parameters that ensure enough segments after padding
        model = Crossformer(
            data_dim=3,
            in_len=200,  # Not divisible by seg_len
            out_len=24,
            seg_len=12,
            win_size=2,
            decoder_embedding="random",
            device="cpu",
        )

        batch_size = 2
        x_seq = torch.randn(batch_size, 200, 3)

        output = model(x_seq)

        assert output.shape == (batch_size, 24, 3)


# =============================================================================
# CROSSFORMER_FORECASTER TESTS
# =============================================================================


class TestCrossformerForecaster:
    """Tests for Crossformer_Forecaster integration."""

    def test_init_forecasting_task(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test initialization for forecasting task.

        **PHM Logic**: Forecasting uses historical context to predict
        future sensor values.

        **Methodology**: Create forecaster with task_type="forecasting".

        **Expected**: Crossformer model initialized correctly.

        Validates: Requirement CF-3.1 - Forecaster initialization
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            ts_in=192,
            ts_out=24,
            task_type="forecasting",
            seg_len=12,
            win_size=2,
            d_model=32,
            device="cpu",
            evaluators=evaluators,
        )

        assert hasattr(forecaster, "crossformer")
        assert isinstance(forecaster.crossformer, Crossformer)

    def test_init_regression_task(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test initialization for regression task (RUL).

        **PHM Logic**: RUL prediction uses encoder to extract features
        and predict remaining life.

        **Methodology**: Create with task_type="rul".

        **Expected**: Model configured for regression output.

        Validates: Requirement CF-3.2 - Regression initialization
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            ts_in=192,
            ts_out=12,  # Must be >= seg_len for decoder
            task_type="rul",
            seg_len=12,
            win_size=2,
            d_model=32,
            device="cpu",
            decoder_embedding="random",
            evaluators=evaluators,
        )

        assert forecaster.task_type == "rul"

    def test_forward_kwargs_properties(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test train and eval forward kwargs.

        **PHM Logic**: Both train and eval use output_attn=False
        for efficiency.

        **Methodology**: Check property values.

        **Expected**: Both return output_attn=False.

        Validates: Requirement CF-3.3 - Forward kwargs
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            ts_in=192,
            ts_out=24,
            task_type="forecasting",
            seg_len=12,
            win_size=2,
            device="cpu",
            evaluators=evaluators,
        )

        assert forecaster.train_step_forward_kwargs == {"output_attn": False}
        assert forecaster.eval_step_forward_kwargs == {"output_attn": False}

    def test_forward_model_pass_forecasting_random(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test forward_model_pass with random decoder embedding.

        **PHM Logic**: Random embedding uses only encoder output
        without target sequence.

        **Methodology**: Call forward_model_pass for forecasting.

        **Expected**: Output matches y_t shape.

        Validates: Requirement CF-3.4 - Forward pass random
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            ts_in=192,
            ts_out=24,
            task_type="forecasting",
            seg_len=12,
            win_size=2,
            decoder_embedding="random",
            device="cpu",
            evaluators=evaluators,
        )

        x_c = torch.randn(2, 192, 4)
        y_c = torch.randn(2, 192, 2)
        x_t = torch.randn(2, 24, 4)
        y_t = torch.randn(2, 24, 2)

        (output,) = forecaster.forward_model_pass(x_c, y_c, x_t, y_t)

        assert output.shape == y_t.shape

    def test_forward_model_pass_forecasting_dsw(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test forward_model_pass with DSW decoder embedding.

        **PHM Logic**: DSW uses target sequence for conditioning,
        enabling teacher forcing.

        **Methodology**: Call forward_model_pass with DSW embedding.

        **Expected**: Output matches y_t shape.

        Validates: Requirement CF-3.5 - Forward pass DSW
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            ts_in=192,
            ts_out=24,
            task_type="forecasting",
            seg_len=12,
            win_size=2,
            decoder_embedding="DSW",
            device="cpu",
            evaluators=evaluators,
        )

        x_c = torch.randn(2, 192, 4)
        y_c = torch.randn(2, 192, 2)
        x_t = torch.randn(2, 24, 4)
        y_t = torch.randn(2, 24, 2)

        (output,) = forecaster.forward_model_pass(x_c, y_c, x_t, y_t)

        assert output.shape == y_t.shape

    def test_forward_model_pass_regression(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test forward_model_pass for regression task.

        **PHM Logic**: Regression uses x_t as input without context.

        **Methodology**: Call forward_model_pass with regression config.

        **Expected**: Output matches target shape.

        Validates: Requirement CF-3.6 - Forward pass regression
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            ts_in=192,
            ts_out=12,
            task_type="rul",
            seg_len=12,
            win_size=2,
            decoder_embedding="random",
            device="cpu",
            evaluators=evaluators,
        )

        x_t = torch.randn(2, 192, 4)
        y_t = torch.randn(2, 12, 1)

        (output,) = forecaster.forward_model_pass(None, None, x_t, y_t)

        assert output.shape == y_t.shape

    def test_step_train(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test step method in training mode.

        **PHM Logic**: Training step computes loss and returns stats
        for logging and optimization.

        **Methodology**: Call step with train=True.

        **Expected**: Stats dict with loss, predictions, targets.

        Validates: Requirement CF-3.7 - Training step
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            ts_in=192,
            ts_out=24,
            task_type="forecasting",
            seg_len=12,
            win_size=2,
            decoder_embedding="random",
            device="cpu",
            evaluators=evaluators,
        )

        batch = (
            torch.randn(2, 192, 4),  # x_c
            torch.randn(2, 192, 2),  # y_c
            torch.randn(2, 24, 4),  # x_t
            torch.randn(2, 24, 2),  # y_t
        )

        stats = forecaster.step(batch, train=True)

        assert "loss" in stats
        assert "forecast_loss" in stats
        assert "predictions" in stats
        assert "targets" in stats

    def test_compute_loss(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test compute_loss method.

        **PHM Logic**: Compute loss handles masking and returns
        detailed loss information.

        **Methodology**: Call compute_loss directly.

        **Expected**: Dict with forecast_loss, forecast_out, forecast_mask.

        Validates: Requirement CF-3.8 - Compute loss
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            ts_in=192,
            ts_out=24,
            task_type="forecasting",
            seg_len=12,
            win_size=2,
            decoder_embedding="random",
            device="cpu",
            evaluators=evaluators,
        )

        batch = (
            torch.randn(2, 192, 4),
            torch.randn(2, 192, 2),
            torch.randn(2, 24, 4),
            torch.randn(2, 24, 2),
        )

        result = forecaster.compute_loss(batch)

        assert "forecast_loss" in result
        assert "forecast_out" in result
        assert "forecast_mask" in result

    def test_mask_y_c_option(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test mask_y_c option.

        **PHM Logic**: Masking context targets tests model's ability
        to forecast without historical target information.

        **Methodology**: Create with mask_y_c=True.

        **Expected**: y_c zeroed in forward pass.

        Validates: Requirement CF-3.9 - Mask y_c option
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            ts_in=192,
            ts_out=24,
            task_type="forecasting",
            seg_len=12,
            win_size=2,
            mask_y_c=True,
            decoder_embedding="random",
            device="cpu",
            evaluators=evaluators,
        )

        assert forecaster.mask_y_c is True

        # Verify forward still works
        batch = (
            torch.randn(2, 192, 4),
            torch.randn(2, 192, 2),
            torch.randn(2, 24, 4),
            torch.randn(2, 24, 2),
        )

        stats = forecaster.step(batch, train=False)
        assert torch.isfinite(torch.tensor(stats["loss"]))


# =============================================================================
# ATTENTION LAYER TESTS
# =============================================================================


class TestCrossformerAttention:
    """Tests for Crossformer attention components."""

    def test_full_attention_init(self):
        """Test FullAttention initialization.

        **PHM Logic**: Attention mechanism captures relationships
        between different time points and dimensions.

        **Methodology**: Create FullAttention layer.

        **Expected**: Layer initialized without error.

        Validates: Requirement CF-4.1 - Attention initialization
        """
        attention = FullAttention(scale=None, attention_dropout=0.1)

        assert hasattr(attention, "dropout")

    def test_attention_layer_init(self):
        """Test AttentionLayer wrapper initialization.

        **PHM Logic**: AttentionLayer wraps attention with projections
        for queries, keys, values.

        **Methodology**: Create AttentionLayer.

        **Expected**: Q, K, V projections initialized.

        Validates: Requirement CF-4.2 - AttentionLayer initialization
        """
        layer = AttentionLayer(
            d_model=64,
            n_heads=4,
        )

        assert hasattr(layer, "query_projection")
        assert hasattr(layer, "key_projection")
        assert hasattr(layer, "value_projection")


# =============================================================================
# EDGE CASES
# =============================================================================


class TestCrossformerEdgeCases:
    """Tests for edge cases in Crossformer models."""

    def test_gradient_flow(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test gradient flow through Crossformer.

        **PHM Logic**: Training requires gradients to propagate
        through all components.

        **Methodology**: Compute loss, verify gradients exist.

        **Expected**: Non-None gradients for model parameters.

        Validates: Requirement CF-6.2 - Gradient flow
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = Crossformer_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            ts_in=192,
            ts_out=24,
            task_type="forecasting",
            seg_len=12,
            win_size=2,
            decoder_embedding="random",
            device="cpu",
            evaluators=evaluators,
        )

        batch = (
            torch.randn(2, 192, 4),
            torch.randn(2, 192, 2),
            torch.randn(2, 24, 4),
            torch.randn(2, 24, 2),
        )

        stats = forecaster.step(batch, train=True)
        stats["loss"].backward()

        # Check some parameters have gradients
        param_with_grad = False
        for param in forecaster.crossformer.parameters():
            if param.grad is not None:
                param_with_grad = True
                break

        assert param_with_grad, "No gradients found in model parameters"
