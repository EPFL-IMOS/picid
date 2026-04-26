"""Tests for CNN1D_Wrapper (explicit signature, regression/classification, forward)."""

import pytest
import torch

from picid.model.estimators.cnn1d.wrapper import CNN1D_Wrapper
from picid.model.estimators.cnn1d.wrapper import _calculate_receptive_field
from picid.model.estimators.cnn1d.model import EncoderModel


def _default_cnn_config(seq_len=7):
    """Config that satisfies receptive field >= seq_len (kernels [4,4] -> receptive field 7)."""
    return {
        "input_channels": 2,
        "latent_dim": 4,
        "dropout_prob": 0.1,
        "output_channels": [4, 4],
        "kernels": [4, 4],
        "strides": [1, 1],
        "dilations": [1, 1],
    }


SEQ_LEN = 7


def test_cnn1d_canonical_modules_are_reported():
    assert EncoderModel.__module__ == "picid.model.estimators.cnn1d.model"


def test_cnn1d_wrapper_init_regression():
    """CNN1D_Wrapper initializes for regression with explicit keyword args."""
    cfg = _default_cnn_config()
    wrapper = CNN1D_Wrapper(
        task_type="rul",
        seq_len=SEQ_LEN,
        **cfg,
    )
    assert wrapper.task_type == "rul"
    assert wrapper.seq_len == SEQ_LEN
    assert wrapper.backbone is not None


def test_cnn1d_wrapper_init_classification():
    """CNN1D_Wrapper initializes for classification when num_classes is provided."""
    cfg = _default_cnn_config()
    wrapper = CNN1D_Wrapper(
        task_type="fault_classification",
        seq_len=SEQ_LEN,
        num_classes=3,
        **cfg,
    )
    assert wrapper.task_type == "fault_classification"
    # EncoderModel final_head is Linear(latent_dim, num_classes)
    assert wrapper.backbone.final_head[0].out_features == 3


def test_cnn1d_wrapper_init_classification_missing_num_classes_raises():
    """CNN1D_Wrapper raises KeyError for classification without num_classes."""
    cfg = _default_cnn_config()
    with pytest.raises(KeyError, match="num_classes"):
        CNN1D_Wrapper(
            task_type="fault_classification",
            seq_len=SEQ_LEN,
            # num_classes omitted
            **cfg,
        )


def test_cnn1d_wrapper_init_unsupported_task_raises():
    """CNN1D_Wrapper raises ValueError for unsupported task_type."""
    cfg = _default_cnn_config()
    with pytest.raises(ValueError, match="not supported"):
        CNN1D_Wrapper(
            task_type="invalid_task",
            seq_len=SEQ_LEN,
            **cfg,
        )


def test_calculate_receptive_field_list_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        _calculate_receptive_field([3], [1, 1], [1])


def test_cnn1d_wrapper_init_seq_len_exceeds_receptive_field_raises():
    """CNN1D_Wrapper raises when seq_len is larger than receptive field."""
    cfg = _default_cnn_config()
    with pytest.raises(ValueError, match="receptive field"):
        CNN1D_Wrapper(
            task_type="rul",
            seq_len=100,
            **cfg,
        )


def test_cnn1d_wrapper_forward_regression():
    """Forward pass for regression returns predictions and targets with expected shapes."""
    cfg = _default_cnn_config()
    wrapper = CNN1D_Wrapper(task_type="rul", seq_len=SEQ_LEN, **cfg)
    batch_size = 4
    batch = {
        "features": torch.randn(batch_size, SEQ_LEN, cfg["input_channels"]),
        "rul": torch.randn(batch_size, 1),
    }
    out = wrapper(batch)
    assert "predictions" in out and "targets" in out
    assert out["predictions"].shape == (batch_size, 1, 1)
    assert out["targets"].shape == (batch_size, 1, 1)


def test_cnn1d_wrapper_forward_classification():
    """Forward pass for classification returns predictions and targets."""
    cfg = _default_cnn_config()
    wrapper = CNN1D_Wrapper(
        task_type="fault_classification",
        seq_len=SEQ_LEN,
        num_classes=2,
        **cfg,
    )
    batch_size = 4
    batch = {
        "features": torch.randn(batch_size, SEQ_LEN, cfg["input_channels"]),
        "fault_classification": torch.randint(0, 2, (batch_size,)),
    }
    out = wrapper(batch)
    assert "predictions" in out and "targets" in out
    assert out["predictions"].shape == (batch_size, 1, 2)
    assert out["targets"].shape == (batch_size, 1)


def test_cnn1d_forward_regression_target_shape_mismatch_raises():
    """Regression path raises when prediction and target tensor shapes disagree."""
    cfg = _default_cnn_config()
    wrapper = CNN1D_Wrapper(task_type="rul", seq_len=SEQ_LEN, **cfg)
    batch = {
        "features": torch.randn(4, SEQ_LEN, cfg["input_channels"]),
        "rul": torch.randn(4, 2),
    }
    with pytest.raises(ValueError, match=r"\[Regression\] Shape mismatch"):
        wrapper(batch)


def test_cnn1d_forward_classification_batch_size_mismatch_raises():
    """Classification path raises when backbone batch dim disagrees with targets."""
    cfg = _default_cnn_config()
    wrapper = CNN1D_Wrapper(
        task_type="fault_classification",
        seq_len=SEQ_LEN,
        num_classes=2,
        **cfg,
    )

    def _short_batch_forward(x):
        return torch.zeros(2, 2, dtype=x.dtype, device=x.device)

    wrapper.backbone.forward = _short_batch_forward  # type: ignore[method-assign]

    batch = {
        "features": torch.randn(4, SEQ_LEN, cfg["input_channels"]),
        "fault_classification": torch.randint(0, 2, (4,)),
    }
    with pytest.raises(ValueError, match=r"\[Classification\] Batch size mismatch"):
        wrapper(batch)


def test_cnn1d_forward_unsupported_task_type_after_corruption_raises():
    """``forward`` final branch when ``task_type`` is not regression/classification."""
    cfg = _default_cnn_config()
    wrapper = CNN1D_Wrapper(task_type="rul", seq_len=SEQ_LEN, **cfg)
    corrupted = "__corrupted_task__"
    wrapper.task_type = corrupted
    batch = {
        "features": torch.randn(2, SEQ_LEN, cfg["input_channels"]),
        corrupted: torch.randn(2, 1),
    }
    with pytest.raises(ValueError, match="Unsupported task type"):
        wrapper(batch)
