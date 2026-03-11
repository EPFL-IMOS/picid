"""Tests for picid.cli.validation."""

from pathlib import Path

from picid.cli.config_discovery import ModelInfo
from picid.cli.validation import (
    filter_models_for_task,
    is_model_compatible_with_task,
)


def test_prognostics_model_compatible_with_prognostics_task():
    """A prognostics model is compatible with a prognostics task."""
    model = ModelInfo(
        task_type="prognostics",
        model="cnn_1d",
        path=Path("configs/model_configs/prognostics/cnn_1d.yaml"),
    )
    assert is_model_compatible_with_task(model, "prognostics") is True


def test_anomaly_model_incompatible_with_prognostics_task():
    """An anomaly_detection model is incompatible with a prognostics task."""
    model = ModelInfo(
        task_type="anomaly_detection",
        model="autoencoder",
        path=Path("configs/model_configs/anomaly_detection/autoencoder.yaml"),
    )
    assert is_model_compatible_with_task(model, "prognostics") is False


def test_filter_models_for_task():
    """filter_models_for_task returns only models matching the task type."""
    models = [
        ModelInfo(
            task_type="prognostics",
            model="cnn_1d",
            path=Path("configs/model_configs/prognostics/cnn_1d.yaml"),
        ),
        ModelInfo(
            task_type="anomaly_detection",
            model="autoencoder",
            path=Path("configs/model_configs/anomaly_detection/autoencoder.yaml"),
        ),
        ModelInfo(
            task_type="prognostics",
            model="transformer",
            path=Path("configs/model_configs/prognostics/transformer.yaml"),
        ),
    ]
    filtered = filter_models_for_task(models, "prognostics")
    assert len(filtered) == 2
    assert all(m.task_type == "prognostics" for m in filtered)
    assert {m.model for m in filtered} == {"cnn_1d", "transformer"}
