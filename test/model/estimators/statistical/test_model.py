import torch
import pytest
import numpy as np

# Adjust imports to match your project structure
from picid.model.estimators.statistical.model import (
    LinearBaseline,
    PolynomialBaseline,
    ExponentialBaseline,
)


def test_statistical_backbones_report_canonical_module():
    assert LinearBaseline.__module__ == "picid.model.estimators.statistical.model"
    assert PolynomialBaseline.__module__ == "picid.model.estimators.statistical.model"
    assert ExponentialBaseline.__module__ == "picid.model.estimators.statistical.model"

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def base_config():
    """Basic config: 2 time steps, 2 channels."""
    return {"seq_len": 2, "input_channels": 2, "poly_degree": 2}


# -------------------------------------------------------------------------
# Deterministic Math Tests
# -------------------------------------------------------------------------


def test_linear_math_correctness(base_config):
    """
    Verifies the Linear Regression math (y = Wx + b) using deterministic weights.

    Logic:
    1. Sets weights to [0.1, 0.2, 0.3, 0.4] and bias to 10.0.
    2. Inputs [1.0, 2.0, 3.0, 4.0].
    3. Expects dot product (0.1*1 + ... + 0.4*4) + 10.0 = 13.0.
    """
    # 1. Init Model
    # Input Dim = 2 * 2 = 4 features
    model = LinearBaseline(base_config, task_type="regression", num_targets=1)

    # 2. Set Interesting Weights
    # Weights: [0.1, 0.2, 0.3, 0.4]
    # Bias: 10.0
    with torch.no_grad():
        w = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        model.linear.weight.data = w
        model.linear.bias.fill_(10.0)

    # 3. Set Input (Batch, Channels, SeqLen)
    # Flattens to: [1, 2, 3, 4]
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    # 4. Manual Calculation
    # (0.1*1) + (0.2*2) + (0.3*3) + (0.4*4) + 10.0
    # = 0.1 + 0.4 + 0.9 + 1.6 + 10.0
    # = 3.0 + 10.0 = 13.0
    expected = 13.0

    # 5. Verify
    out = model(x)
    assert torch.isclose(out, torch.tensor([expected])).all()


def test_linear_zeros_logic(base_config):
    """
    Verifies that passing a zero tensor results in an output exactly equal to the bias.

    Logic:
    1. Sets random weights and a known bias (2.0).
    2. Inputs a tensor of zeros.
    3. Expects Output = 0 * Weights + Bias = 2.0.
    """
    model = LinearBaseline(base_config, task_type="regression", num_targets=1)

    with torch.no_grad():
        model.linear.weight.fill_(5.5)  # Random non-zero weight
        model.linear.bias.fill_(2.0)

    # Input is all zeros
    x = torch.zeros(1, 2, 2)

    # Output should be just the bias
    out = model(x)
    assert torch.isclose(out, torch.tensor([2.0])).all()


def test_polynomial_math_correctness(base_config):
    """
    Verifies Polynomial Regression feature expansion and calculation.

    Logic:
    1. Configures degree=2 (Linear + Quadratic).
    2. Sets linear weight=0.5, quadratic weight=0.1, bias=1.0.
    3. Inputs x=4.0.
    4. Expects y = (0.5 * 4) + (0.1 * 4^2) + 1.0 = 2.0 + 1.6 + 1.0 = 4.6.
    """
    config = {"seq_len": 1, "input_channels": 1, "poly_degree": 2}
    model = PolynomialBaseline(config, task_type="regression", num_targets=1)

    # 2. Set Weights for [x, x^2]
    # w_lin = 0.5, w_quad = 0.1, Bias = 1.0
    with torch.no_grad():
        model.linear.weight.data = torch.tensor([[0.5, 0.1]])
        model.linear.bias.fill_(1.0)

    # 3. Input x = 4.0
    x = torch.tensor([[[4.0]]])

    # 4. Manual Calculation
    expected = 4.6

    # 5. Verify
    out = model(x)
    assert torch.isclose(out, torch.tensor([expected])).all()


def test_exponential_math_correctness(base_config):
    """
    Verifies Exponential Regression math y = exp(Wx + b).

    Logic:
    1. Case A: Weights=1, Bias=0, Input=1. Expects exp(1).
    2. Case B: Bias=2, Input=0. Expects exp(2).
    """
    config = {"seq_len": 1, "input_channels": 1}
    model = ExponentialBaseline(config, task_type="regression", num_targets=1)

    # Case 1: Simple e^1
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)

    x = torch.tensor([[[1.0]]])
    out = model(x)

    # Cast numpy result to float32 to match model output type
    expected = torch.tensor([np.exp(1.0)], dtype=torch.float)
    assert torch.isclose(out, expected).all()

    # Case 2: Zero Input -> e^b
    with torch.no_grad():
        model.linear.bias.fill_(2.0)  # bias = 2

    x_zero = torch.tensor([[[0.0]]])
    out_zero = model(x_zero)
    # y = exp(1.0*0 + 2.0) = exp(2)

    # Cast numpy result to float32
    expected_zero = torch.tensor([np.exp(2.0)], dtype=torch.float)
    assert torch.isclose(out_zero, expected_zero).all()


def test_classification_logits_deterministic(base_config):
    """
    Verifies that the Linear model correctly separates weights for multi-class outputs.

    Logic:
    1. Sets up a 2-class problem.
    2. Class 0: Weight=1.0, Bias=0.0.
    3. Class 1: Weight=-1.0, Bias=5.0.
    4. Input x=10.0.
    5. Expects Logits: [10.0, -5.0].
    """
    # 1. Init Model for Classification (2 Classes)
    config = {"seq_len": 1, "input_channels": 1}

    # Use 'num_targets' as the generic output dimension for the backbone
    model = LinearBaseline(config, task_type="classification", num_targets=2)

    # 2. Set Deterministic Weights
    # Weights shape: (Num_Classes, Input_Dim) -> (2, 1)
    with torch.no_grad():
        model.linear.weight.data = torch.tensor([[1.0], [-1.0]])
        model.linear.bias.data = torch.tensor([0.0, 5.0])

    # 3. Create Input
    x = torch.tensor([[[10.0]]])  # (Batch=1, Ch=1, Seq=1)

    # 4. Forward Pass
    logits = model(x)

    # 5. Expected Output
    expected_logits = torch.tensor([[10.0, -5.0]])

    assert torch.allclose(logits, expected_logits)
    assert logits.shape == (1, 2)


def test_exponential_classification_error(base_config):
    """
    Verifies that ExponentialBaseline raises a ValueError if instantiated for classification.
    Target: Covers the 'if task_type == "classification": raise ValueError' block.
    """
    with pytest.raises(ValueError, match="does not support Classification"):
        ExponentialBaseline(base_config, task_type="classification", num_targets=2)
