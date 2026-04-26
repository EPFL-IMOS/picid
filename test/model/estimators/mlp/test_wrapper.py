import torch
import torch.nn as nn
import pytest
import logging
from picid.model.estimators.mlp.model import MLP
from picid.model.estimators.mlp.wrapper import MLPWrapper

# =========================================================================
# 1. MLP Model Logic Coverage (Targeting image_10ea2a.png)
# =========================================================================


def test_mlp_now_reports_canonical_module():
    assert MLP.__module__ == "picid.model.estimators.mlp.model"


def test_mlp_model_validation_invalid_layers():
    """
    [Coverage: MLP Lines 44-45]
    Verifies that initializing MLP with `num_layers < 1` raises a ValueError.
    """
    config = {"seq_len": 10, "input_channels": 2, "num_layers": 0}
    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        MLP(config, task_type="regression", num_targets=1)


def test_mlp_model_single_layer_path():
    """
    [Coverage: MLP Lines 48-50]
    Verifies the `if self.num_layers == 1` block.
    The network should contain exactly one Linear layer and no activations/norm.
    """
    config = {"seq_len": 5, "input_channels": 2, "num_layers": 1}
    model = MLP(config, task_type="regression", num_targets=1)

    # Expecting simple Linear Map: In -> Out
    assert len(model.net) == 1
    assert isinstance(model.net[0], nn.Linear)
    # Input dim (5*2=10) -> Output dim (1)
    assert model.net[0].in_features == 10
    assert model.net[0].out_features == 1


def test_mlp_model_standard_two_layers():
    """
    [Coverage: MLP Lines 52-60 (Else Block)]
    Verifies the default path (num_layers=2).
    Structure: [Linear->LN->ReLU] -> [Linear]
    Total modules: 3 (in first block) + 1 (final) = 4.
    """
    config = {"seq_len": 5, "input_channels": 2, "num_layers": 2, "hidden_dim": 8}
    model = MLP(config, task_type="regression", num_targets=1)

    assert len(model.net) == 4
    # First layer is Linear(10 -> 8)
    assert model.net[0].in_features == 10
    assert model.net[0].out_features == 8


def test_mlp_model_deep_layers_loop():
    """
    [Coverage: MLP Lines 64-69 (Loop logic)]
    Verifies the `for _ in range(self.num_layers - 2)` block.
    We set num_layers=4 to force the loop to run twice.
    Structure:
      1. Input Block (3 modules)
      2. Loop Block 1 (3 modules)
      3. Loop Block 2 (3 modules)
      4. Final Linear (1 module)
    Total: 10 modules.
    """
    config = {"seq_len": 5, "input_channels": 2, "num_layers": 4, "hidden_dim": 8}
    model = MLP(config, task_type="regression", num_targets=1)

    assert len(model.net) == 10

    # Verify execution flow
    x = torch.randn(2, 2, 5)  # (Batch, Chan, Seq)
    out = model(x)
    assert out.shape == (2, 1)


def test_mlp_initialization_logic():
    """
    Verifies that weights are initialized non-default.
    - Linear Bias should be 0.
    - LayerNorm Weight should be 1.0.
    - LayerNorm Bias should be 0.0.
    """
    config = {"seq_len": 2, "input_channels": 2, "num_layers": 2}
    model = MLP(config, task_type="regression", num_targets=1)

    # Check LayerNorm (index 1)
    ln = model.net[1]
    assert torch.allclose(ln.weight, torch.ones_like(ln.weight))
    assert torch.allclose(ln.bias, torch.zeros_like(ln.bias))

    # Check Linear Bias (index 0)
    lin = model.net[0]
    assert torch.allclose(lin.bias, torch.zeros_like(lin.bias))


# =========================================================================
# 2. Wrapper Logic Coverage (Targeting image_10e9ad.jpg)
# =========================================================================


def test_wrapper_init_unsupported_task():
    """
    [Coverage: Wrapper Lines 33-34]
    Verifies the `if task_type not in self.supported_types` check.
    """
    with pytest.raises(ValueError, match="is not supported"):
        MLPWrapper(task_type="invalid_task_name", seq_len=10, input_channels=2)


def test_wrapper_init_regression_legacy_args():
    """
    [Coverage: Wrapper Lines 44-46]
    Verifies the `elif "output_dim" in kwargs` block.
    The wrapper should accept `output_dim` if `num_targets` is missing.
    """
    task = "rul"
    wrapper = MLPWrapper(
        task_type=task,
        seq_len=10,
        input_channels=2,
        output_dim=5,  # Legacy kwarg
    )
    assert wrapper.backbone.output_dim == 5


def test_wrapper_init_classification_missing_args():
    """
    [Coverage: Wrapper Lines 52-54]
    Verifies `if "num_classes" not in kwargs` raises KeyError for classification.
    """
    task = "fault_classification"
    with pytest.raises(KeyError, match="num_classes' must be provided"):
        MLPWrapper(
            task_type=task,
            seq_len=10,
            input_channels=2,
            # Missing num_classes
        )


def test_wrapper_init_missing_channels():
    """
    Verifies that omitting required input_channels raises (TypeError for missing
    keyword-only argument, or KeyError for legacy kwargs-based API).
    """
    task = "rul"
    with pytest.raises((TypeError, KeyError), match="input_channels"):
        MLPWrapper(
            task_type=task,
            seq_len=10,
            num_targets=1,
            # Missing input_channels
        )


def test_wrapper_forward_regression_mismatch_logging(caplog):
    """
    [Coverage: Wrapper Lines 88-90]
    Verifies regression shape check logic.
    If predictions.shape[-1] != targets.shape[-1], it must log a DEBUG message.
    """
    task = "rul"
    wrapper = MLPWrapper(task_type=task, seq_len=10, input_channels=2, num_targets=1)

    # Mismatch: Prediction (B, 1, 1) vs Target (B, 2)
    # The wrapper converts target to (B, 1, 2) -> Mismatch on last dim
    batch = {"features": torch.randn(2, 10, 2), "rul": torch.randn(2, 2)}

    with caplog.at_level(logging.DEBUG):
        # We invoke forward directly or via call.
        # Note: Depending on base class, this might crash on subsequent assertions,
        # but we only care that the log was emitted first.
        try:
            wrapper(batch)
        except Exception:
            pass  # Ignore subsequent crashes (like shape assertion in base)

    assert "Dimension mismatch" in caplog.text


def test_wrapper_forward_classification_reshaping():
    """
    [Coverage: Wrapper Lines 95-96]
    Verifies classification reshaping logic.
    - Preds: (B, C) -> (B, 1, C)
    - Targets: (B,) -> (B, 1)
    """
    task = "fault_classification"
    wrapper = MLPWrapper(task_type=task, seq_len=10, input_channels=2, num_classes=3)

    # Batch size 2
    batch = {
        "features": torch.randn(2, 10, 2),
        "fault_classification": torch.tensor([0, 1]),  # Shape (2,)
    }

    out = wrapper(batch)
    preds = out["predictions"]
    targets = out["targets"]

    assert preds.shape == (2, 1, 3)  # (Batch, 1, Classes)
    assert targets.shape == (2, 1)  # (Batch, 1)


def test_wrapper_forward_corrupted_task_failsafe():
    """
    [Coverage: Wrapper Lines 98-99]
    Verifies the final `else` block in forward().
    This handles the theoretical case where task_type becomes invalid after init.
    """
    task = "rul"
    wrapper = MLPWrapper(task_type=task, seq_len=10, input_channels=2, num_targets=1)

    # Manually corrupt the task type
    wrapper.task_type = "corrupted_task"
    batch = {"features": torch.randn(2, 10, 2), "corrupted_task": torch.randn(2, 1)}

    with pytest.raises(ValueError, match="Unsupported task type"):
        wrapper(batch)
