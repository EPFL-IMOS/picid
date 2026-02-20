"""Tests for CNN1D_Wrapper (explicit signature, regression/classification, forward)."""
import pytest
import torch

from picid.model.wrappers.cnn1d_wrapper import CNN1D_Wrapper


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
