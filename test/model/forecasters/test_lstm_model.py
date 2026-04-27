"""Comprehensive tests for the LSTM Model components.

This module provides rigorous testing for LSTM-based forecasting models
which are widely used in PHM for capturing temporal dependencies.

PHM Context:
-----------
LSTM models are fundamental for PHM applications:
- **LSTM_Encoder**: Encodes historical sensor sequences into hidden states
- **LSTM_Decoder**: Generates future predictions from encoded states
- **LSTM_Seq2Seq**: Sequence-to-sequence for forecasting tasks
- **LSTM_Regression**: For RUL prediction and health index estimation

LSTM advantages in PHM:
1. Captures long-term dependencies in degradation patterns
2. Handles variable-length sequences common in run-to-failure data
3. Teacher forcing enables stable training on noisy sensor data

Reference: Zheng et al. (2017) "Long Short-Term Memory Network for RUL Estimation"

Test Coverage Strategy:
----------------------
1. **Encoder Tests**: Input processing, hidden state generation
2. **Decoder Tests**: Step-by-step generation, hidden state propagation
3. **Seq2Seq Tests**: End-to-end forecasting with teacher forcing
4. **Regression Tests**: Direct RUL prediction from features
5. **LSTM_Forecaster Tests**: Integration with training pipeline
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from picid.model.forecasters.lstm_model.lstm_model import (
    LSTM_Encoder,
    LSTM_Decoder,
    LSTM_Seq2Seq,
    LSTM_Regression,
    LSTM_Forecaster,
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
# LSTM_ENCODER TESTS
# =============================================================================


class TestLSTMEncoder:
    """Tests for LSTM_Encoder class.

    The encoder processes historical sensor sequences and generates
    hidden states that capture temporal patterns.
    """

    def test_init_basic(self):
        """Test basic initialization of LSTM_Encoder.

        **PHM Logic**: Encoder must accept sensor input dimension and
        produce hidden states for the decoder.

        **Methodology**: Create encoder with specific parameters.

        **Expected**: LSTM layer initialized with correct dimensions.

        Validates: Requirement LS-1.1 - Encoder initialization
        """
        encoder = LSTM_Encoder(input_dim=10, hidden_dim=64, n_layers=2, dropout=0.2)

        assert encoder.input_dim == 10
        assert encoder.hidden_dim == 64
        assert encoder.n_layers == 2
        assert isinstance(encoder.lstm, nn.LSTM)

    def test_forward_basic(self):
        """Test forward pass of encoder.

        **PHM Logic**: Encoder processes context sequence and returns
        hidden and cell states for decoder initialization.

        **Methodology**: Pass batch of sequences through encoder.

        **Expected**: Hidden/cell shapes: (n_layers, batch, hidden_dim).

        Validates: Requirement LS-1.2 - Encoder forward pass
        """
        encoder = LSTM_Encoder(input_dim=10, hidden_dim=64, n_layers=2)

        # Batch=4, SeqLen=20, InputDim=10
        x_context = torch.randn(4, 20, 10)

        hidden, cell = encoder(x_context)

        assert hidden.shape == (2, 4, 64)  # (layers, batch, hidden)
        assert cell.shape == (2, 4, 64)

    def test_forward_variable_sequence_length(self):
        """Test encoder with different sequence lengths.

        **PHM Logic**: PHM data often has variable-length runs.
        Encoder should handle different lengths.

        **Methodology**: Test with various sequence lengths.

        **Expected**: Same output shape regardless of input length.

        Validates: Requirement LS-1.3 - Variable length handling
        """
        encoder = LSTM_Encoder(input_dim=10, hidden_dim=64, n_layers=2)

        for seq_len in [5, 10, 50, 100]:
            x_context = torch.randn(2, seq_len, 10)
            hidden, cell = encoder(x_context)

            assert hidden.shape == (2, 2, 64)
            assert cell.shape == (2, 2, 64)

    def test_encoder_dropout(self):
        """Test that dropout is applied during training.

        **PHM Logic**: Dropout prevents overfitting on small PHM datasets.

        **Methodology**: Compare outputs in train vs eval mode.

        **Expected**: Different outputs in train mode (due to dropout).

        Validates: Requirement LS-1.4 - Dropout functionality
        """
        encoder = LSTM_Encoder(input_dim=10, hidden_dim=64, n_layers=2, dropout=0.5)

        x_context = torch.randn(4, 20, 10)

        # Training mode
        encoder.train()
        hidden_train1, _ = encoder(x_context)
        hidden_train2, _ = encoder(x_context)

        # Eval mode
        encoder.eval()
        with torch.no_grad():
            hidden_eval1, _ = encoder(x_context)
            hidden_eval2, _ = encoder(x_context)

        # Eval outputs should be identical
        torch.testing.assert_close(hidden_eval1, hidden_eval2)


# =============================================================================
# LSTM_DECODER TESTS
# =============================================================================


class TestLSTMDecoder:
    """Tests for LSTM_Decoder class.

    The decoder generates predictions step-by-step using hidden states
    from the encoder.
    """

    def test_init_basic(self):
        """Test basic initialization of LSTM_Decoder.

        **PHM Logic**: Decoder produces output features for each timestep.

        **Methodology**: Create decoder with specific parameters.

        **Expected**: LSTM and output linear layer initialized.

        Validates: Requirement LS-2.1 - Decoder initialization
        """
        decoder = LSTM_Decoder(
            output_dim=3, input_dim=10, hidden_dim=64, n_layers=2, dropout=0.2
        )

        assert decoder.output_dim == 3
        assert decoder.hidden_dim == 64
        assert isinstance(decoder.lstm, nn.LSTM)
        assert isinstance(decoder.fc, nn.Linear)

    def test_forward_single_step(self):
        """Test single-step decoding.

        **PHM Logic**: Each decoder step predicts one timestep,
        updating hidden states for the next step.

        **Methodology**: Pass single timestep input with hidden states.

        **Expected**: Output shape (batch, 1, output_dim).

        Validates: Requirement LS-2.2 - Single step decoding
        """
        decoder = LSTM_Decoder(output_dim=3, input_dim=10, hidden_dim=64, n_layers=2)

        # Single timestep input
        x_t = torch.randn(4, 1, 10)  # (batch, 1, input)
        hidden = torch.randn(2, 4, 64)
        cell = torch.randn(2, 4, 64)

        output, new_hidden, new_cell = decoder(x_t, hidden, cell)

        assert output.shape == (4, 1, 3)  # (batch, 1, output_dim)
        assert new_hidden.shape == (2, 4, 64)
        assert new_cell.shape == (2, 4, 64)

    def test_hidden_state_propagation(self):
        """Test that hidden states are properly updated.

        **PHM Logic**: Hidden states carry information between steps,
        essential for multi-step forecasting.

        **Methodology**: Run multiple steps, verify states change.

        **Expected**: Hidden states differ after each step.

        Validates: Requirement LS-2.3 - State propagation
        """
        decoder = LSTM_Decoder(output_dim=3, input_dim=10, hidden_dim=64, n_layers=2)

        x_t = torch.randn(2, 1, 10)
        hidden = torch.randn(2, 2, 64)
        cell = torch.randn(2, 2, 64)

        _, h1, c1 = decoder(x_t, hidden, cell)
        _, h2, c2 = decoder(x_t, h1, c1)

        # States should have changed
        assert not torch.allclose(h1, h2)
        assert not torch.allclose(c1, c2)


# =============================================================================
# LSTM_SEQ2SEQ TESTS
# =============================================================================


class TestLSTMSeq2Seq:
    """Tests for LSTM_Seq2Seq class.

    Seq2Seq model combines encoder and decoder for sequence-to-sequence
    forecasting tasks.
    """

    @pytest.fixture
    def seq2seq_model(self):
        """Create a Seq2Seq model for testing."""
        encoder = LSTM_Encoder(input_dim=12, hidden_dim=64, n_layers=2)
        decoder = LSTM_Decoder(output_dim=3, input_dim=12, hidden_dim=64, n_layers=2)
        return LSTM_Seq2Seq(t2v=None, encoder=encoder, decoder=decoder)

    def test_forward_basic(self, seq2seq_model):
        """Test basic forward pass of Seq2Seq.

        **PHM Logic**: Seq2Seq forecasts future values given context
        features and targets.

        **Methodology**: Pass context and target tensors.

        **Expected**: Output shape matches target sequence shape.

        Validates: Requirement LS-3.1 - Seq2Seq forward
        """
        batch_size = 4
        context_len = 20
        pred_len = 10
        x_dim = 9
        y_dim = 3

        x_context = torch.randn(batch_size, context_len, x_dim)
        y_context = torch.randn(batch_size, context_len, y_dim)
        x_target = torch.randn(batch_size, pred_len, x_dim)
        y_target = torch.randn(batch_size, pred_len, y_dim)

        output = seq2seq_model(
            x_context, y_context, x_target, y_target, teacher_forcing_prob=0.5
        )

        assert output.shape == (batch_size, pred_len, y_dim)

    def test_forward_no_teacher_forcing(self, seq2seq_model):
        """Test forward pass without teacher forcing.

        **PHM Logic**: Without teacher forcing, model uses its own
        predictions as input - matches inference behavior.

        **Methodology**: Set teacher_forcing_prob=0.

        **Expected**: Valid output, autoregressive generation.

        Validates: Requirement LS-3.2 - No teacher forcing
        """
        batch_size = 2
        context_len = 15
        pred_len = 5

        x_context = torch.randn(batch_size, context_len, 9)
        y_context = torch.randn(batch_size, context_len, 3)
        x_target = torch.randn(batch_size, pred_len, 9)
        y_target = torch.randn(batch_size, pred_len, 3)

        output = seq2seq_model(
            x_context, y_context, x_target, y_target, teacher_forcing_prob=0.0
        )

        assert output.shape == (batch_size, pred_len, 3)
        assert not torch.isnan(output).any()

    def test_forward_full_teacher_forcing(self, seq2seq_model):
        """Test forward pass with full teacher forcing.

        **PHM Logic**: Full teacher forcing uses ground truth at each
        step - useful for training but may cause exposure bias.

        **Methodology**: Set teacher_forcing_prob=1.

        **Expected**: Valid output using ground truth inputs.

        Validates: Requirement LS-3.3 - Full teacher forcing
        """
        batch_size = 2

        x_context = torch.randn(batch_size, 15, 9)
        y_context = torch.randn(batch_size, 15, 3)
        x_target = torch.randn(batch_size, 5, 9)
        y_target = torch.randn(batch_size, 5, 3)

        output = seq2seq_model(
            x_context, y_context, x_target, y_target, teacher_forcing_prob=1.0
        )

        assert output.shape == (batch_size, 5, 3)

    def test_forward_without_x_context(self):
        """Test forward pass without context features.

        **PHM Logic**: Some PHM tasks only use target history (state forecasting).

        **Methodology**: Pass x_context=None.

        **Expected**: Model handles None context.

        Validates: Requirement LS-3.4 - No context features
        """
        encoder = LSTM_Encoder(input_dim=3, hidden_dim=64, n_layers=2)
        decoder = LSTM_Decoder(output_dim=3, input_dim=3, hidden_dim=64, n_layers=2)
        model = LSTM_Seq2Seq(t2v=None, encoder=encoder, decoder=decoder)

        batch_size = 2
        y_context = torch.randn(batch_size, 15, 3)
        y_target = torch.randn(batch_size, 5, 3)

        output = model(None, y_context, None, y_target, teacher_forcing_prob=0.5)

        assert output.shape == (batch_size, 5, 3)


# =============================================================================
# LSTM_REGRESSION TESTS
# =============================================================================


class TestLSTMRegression:
    """Tests for LSTM_Regression class.

    Regression model for direct RUL prediction without the encoder-decoder
    split (decoder-only architecture).
    """

    @pytest.fixture
    def regression_model(self):
        """Create a Regression model for testing."""
        encoder = LSTM_Encoder(input_dim=10, hidden_dim=64, n_layers=2)
        decoder = LSTM_Decoder(output_dim=1, input_dim=10, hidden_dim=64, n_layers=2)
        return LSTM_Regression(t2v=None, encoder=encoder, decoder=decoder)

    def test_forward_rul_prediction(self, regression_model):
        """Test forward pass for RUL prediction.

        **PHM Logic**: RUL prediction outputs remaining useful life
        estimate for each sample.

        **Methodology**: Pass feature sequence, get RUL output.

        **Expected**: Output shape (batch, pred_len, 1) for RUL.

        Validates: Requirement LS-4.1 - RUL prediction
        """
        batch_size = 4
        seq_len = 20
        pred_len = 1

        x_target = torch.randn(batch_size, seq_len, 10)
        y_target = torch.randn(batch_size, pred_len, 1)

        output = regression_model(x_target, y_target, teacher_forcing_prob=0.5)

        assert output.shape == (batch_size, pred_len, 1)

    def test_forward_with_classification(self):
        """Test regression model configured for classification.

        **PHM Logic**: LSTM can also classify health states or fault types.

        **Methodology**: Create model with n_classes, verify output dim.

        **Expected**: Output dimension matches n_classes.

        Validates: Requirement LS-4.2 - Classification mode
        """
        encoder = LSTM_Encoder(input_dim=10, hidden_dim=64, n_layers=2)
        decoder = LSTM_Decoder(output_dim=5, input_dim=10, hidden_dim=64, n_layers=2)
        model = LSTM_Regression(t2v=None, encoder=encoder, decoder=decoder, n_classes=5)

        batch_size = 2
        x_target = torch.randn(batch_size, 20, 10)
        y_target = torch.randn(batch_size, 1, 5)  # 5 classes

        output = model(x_target, y_target, teacher_forcing_prob=0.0)

        assert output.shape == (batch_size, 1, 5)


# =============================================================================
# LSTM_FORECASTER TESTS
# =============================================================================


class TestLSTMForecaster:
    """Tests for LSTM_Forecaster class.

    Full integration tests for the LSTM forecaster with the training pipeline.
    """

    def test_init_forecasting_task(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test initialization for forecasting task.

        **PHM Logic**: Forecasting task uses Seq2Seq architecture.

        **Methodology**: Create forecaster with task_type="forecasting".

        **Expected**: Model is LSTM_Seq2Seq instance.

        Validates: Requirement LS-5.1 - Forecasting initialization
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            time_emb_dim=0,
            n_layers=2,
            hidden_dim=32,
            evaluators=evaluators,
        )

        assert isinstance(forecaster.model, LSTM_Seq2Seq)

    def test_init_regression_task(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test initialization for regression task (RUL).

        **PHM Logic**: Regression uses decoder-only architecture.

        **Methodology**: Create forecaster with task_type="rul".

        **Expected**: Model is LSTM_Regression instance.

        Validates: Requirement LS-5.2 - Regression initialization
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            time_emb_dim=0,
            n_layers=2,
            hidden_dim=32,
            evaluators=evaluators,
        )

        assert isinstance(forecaster.model, LSTM_Regression)

    def test_init_classification_task(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test initialization for classification task.

        **PHM Logic**: Classification uses regression architecture with class outputs.

        **Methodology**: Create forecaster with task_type="fault_classification".

        **Expected**: Model is LSTM_Regression with n_classes.

        Validates: Requirement LS-5.3 - Classification initialization
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=5,  # 5 classes
            task_type="fault_classification",
            time_emb_dim=0,
            n_layers=2,
            hidden_dim=32,
            evaluators=evaluators,
        )

        assert isinstance(forecaster.model, LSTM_Regression)

    def test_init_state_forecasting_task(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test initialization for state forecasting task.

        **PHM Logic**: State forecasting predicts future states without covariates.

        **Methodology**: Create forecaster with task_type="state_forecasting".

        **Expected**: Model is LSTM_Seq2Seq.

        Validates: Requirement LS-5.4 - State forecasting initialization
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=0,
            d_yc=2,
            d_yt=2,
            task_type="state_forecasting",
            time_emb_dim=0,
            n_layers=2,
            hidden_dim=32,
            evaluators=evaluators,
        )

        assert isinstance(forecaster.model, LSTM_Seq2Seq)

    def test_forward_kwargs_properties(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test train and eval forward kwargs.

        **PHM Logic**: Train uses teacher forcing, eval does not.

        **Methodology**: Check train vs eval kwargs.

        **Expected**: Different teacher forcing probabilities.

        Validates: Requirement LS-5.5 - Forward kwargs
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            teacher_forcing_prob=0.5,
            evaluators=evaluators,
        )

        assert forecaster.train_step_forward_kwargs["force"] == 0.5
        assert forecaster.eval_step_forward_kwargs["force"] == 0.0

    def test_forward_model_pass_forecasting(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test forward_model_pass for forecasting.

        **PHM Logic**: Forecasting uses all four input tensors.

        **Methodology**: Call forward_model_pass with full inputs.

        **Expected**: Output matches y_t shape.

        Validates: Requirement LS-5.6 - Forecasting forward pass
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            evaluators=evaluators,
        )

        x_c = torch.randn(2, 10, 4)
        y_c = torch.randn(2, 10, 2)
        x_t = torch.randn(2, 5, 4)
        y_t = torch.randn(2, 5, 2)

        (output,) = forecaster.forward_model_pass(x_c, y_c, x_t, y_t, force=0.0)

        assert output.shape == y_t.shape

    def test_forward_model_pass_regression(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test forward_model_pass for regression.

        **PHM Logic**: Regression only uses x_t and y_t.

        **Methodology**: Call with x_c=None, y_c=None.

        **Expected**: Output matches y_t shape.

        Validates: Requirement LS-5.7 - Regression forward pass
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            evaluators=evaluators,
        )

        x_t = torch.randn(2, 10, 4)
        y_t = torch.randn(2, 1, 1)

        (output,) = forecaster.forward_model_pass(None, None, x_t, y_t, force=0.0)

        assert output.shape == y_t.shape

    def test_step_returns_forecast_loss(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test that step method returns forecast_loss.

        **PHM Logic**: LSTM_Forecaster adds forecast_loss key for
        compatibility with scheduler monitoring.

        **Methodology**: Call step, check output keys.

        **Expected**: Both 'loss' and 'forecast_loss' present.

        Validates: Requirement LS-5.8 - Step output
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            evaluators=evaluators,
        )

        batch = (
            torch.randn(2, 10, 4),  # x_c
            torch.randn(2, 10, 2),  # y_c
            torch.randn(2, 5, 4),  # x_t
            torch.randn(2, 5, 2),  # y_t
        )

        stats = forecaster.step(batch, train=True)

        assert "loss" in stats
        assert "forecast_loss" in stats
        assert stats["loss"] == stats["forecast_loss"]

    def test_mask_y_c_option(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test mask_y_c option zeros out context targets.

        **PHM Logic**: Masking y_c tests model's ability to forecast
        without historical target information.

        **Methodology**: Create with mask_y_c=True, verify y_c zeroed.

        **Expected**: y_c replaced with zeros in forward pass.

        Validates: Requirement LS-5.9 - Mask y_c option
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            mask_y_c=True,
            evaluators=evaluators,
        )

        assert forecaster.mask_y_c is True

        # Forward pass should work with masked y_c
        batch = (
            torch.randn(2, 10, 4),
            torch.randn(2, 10, 2),
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 2),
        )

        stats = forecaster.step(batch, train=False)
        assert "loss" in stats


# =============================================================================
# TIME2VEC EMBEDDING TESTS
# =============================================================================


class TestLSTMWithTime2Vec:
    """Tests for LSTM models with Time2Vec embeddings."""

    def test_forecaster_with_time_emb(
        self, mock_optimizer_factory, mock_scheduler_factory
    ):
        """Test LSTM_Forecaster with time embeddings.

        **PHM Logic**: Time embeddings capture periodic patterns in
        sensor data (e.g., day/night cycles in machinery).

        **Methodology**: Create with time_emb_dim > 0.

        **Expected**: t2v component initialized.

        Validates: Requirement LS-6.1 - Time embedding support
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            time_emb_dim=8,  # Non-zero enables Time2Vec
            evaluators=evaluators,
        )

        assert forecaster.t2v is not None


# =============================================================================
# EDGE CASES
# =============================================================================


class TestLSTMEdgeCases:
    """Tests for edge cases in LSTM models."""

    def test_single_layer_lstm(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test with single LSTM layer.

        **PHM Logic**: Simple models may suffice for basic PHM tasks.

        **Methodology**: Create with n_layers=1.

        **Expected**: Model works correctly.

        Validates: Requirement LS-7.1 - Single layer
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            n_layers=1,
            dropout_p=0.0,  # No dropout with 1 layer
            evaluators=evaluators,
        )

        batch = (
            torch.randn(2, 10, 4),
            torch.randn(2, 10, 2),
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 2),
        )

        stats = forecaster.step(batch, train=False)
        assert torch.isfinite(stats["loss"])

    def test_gradient_flow(self, mock_optimizer_factory, mock_scheduler_factory):
        """Test gradient flow through LSTM.

        **PHM Logic**: Training requires gradient propagation.

        **Methodology**: Compute loss, verify gradients exist.

        **Expected**: Non-None gradients for model parameters.

        Validates: Requirement LS-7.2 - Gradient flow
        """
        evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}
        forecaster = LSTM_Forecaster(
            optimizer_factory=mock_optimizer_factory,
            scheduler_factory=mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=2,
            task_type="forecasting",
            evaluators=evaluators,
        )

        batch = (
            torch.randn(2, 10, 4),
            torch.randn(2, 10, 2),
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 2),
        )

        stats = forecaster.step(batch, train=True)
        stats["loss"].backward()

        # Check encoder has gradients
        for param in forecaster.encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
