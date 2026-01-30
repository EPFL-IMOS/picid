import pytest
import xarray as xr
import numpy as np
from pathlib import Path
from lmetk.evaluator.default import DefaultEvaluator
from lmetk.evaluator.classification import ClassificationEvaluator
from lmetk.evaluator.forecasting import ForecastingEvaluator
from lmetk.evaluator.reconstruction import ReconstructionEvaluator


@pytest.fixture
def temp_output_dir(tmp_path):
    """Creates a temporary root for artifact validation."""
    return tmp_path / "outputs"


def test_classification_netcdf_schema(temp_output_dir):
    """
    Classification Audit: Ensures class indices and probabilities
    are mapped correctly.
    """
    evaluator = ClassificationEvaluator(
        metric_names=["accuracy"],
        num_classes=3,
        save_predictions=True,
        paths={"eval_details": str(temp_output_dir)},
    )

    # Simulate 3 batches of 4 samples
    for _ in range(3):
        evaluator.update(
            {
                "predictions": np.random.rand(4, 3),  # Probabilities
                "targets": np.random.randint(0, 3, (4,)),  # Class indices
            }
        )

    evaluator.compute(mode="test", epoch=5, step=100)

    file_path = temp_output_dir / "test" / "predictions_epoch_5.nc"
    with xr.open_dataset(file_path) as ds:
        # Classification usually has (sample, class) for preds
        # and (sample) for targets
        assert "sample" in ds.dims
        assert "class" in ds.dims
        assert ds.preds.shape == (12, 3)
        assert ds.targets.shape == (12,)
        # Verify metadata
        assert ds.attrs["task_type"] == "classification"


def test_default_regression_netcdf_schema(temp_output_dir):
    """
    Default Audit: Standard regression schema check.
    """
    evaluator = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        save_predictions=True,
        paths={"eval_details": str(temp_output_dir)},
    )

    # 1D Regression (Batch=5, Output=1)
    evaluator.update(
        {"predictions": np.random.rand(5, 1), "targets": np.random.rand(5, 1)}
    )
    evaluator.compute(mode="val", epoch=2, step=0)

    file_path = temp_output_dir / "val" / "predictions_epoch_2.nc"
    with xr.open_dataset(file_path) as ds:
        assert "sample" in ds.dims
        assert "feature" in ds.dims
        assert ds.preds.shape == (5, 1)


def test_forecasting_netcdf_schema(temp_output_dir):
    """
    Forecasting Audit: Checks for 'horizon' dimension.
    """
    evaluator = ForecastingEvaluator(
        metric_names=["mse"],
        model_pred_len=24,
        effective_pred_len=12,
        save_predictions=True,
        paths={"eval_details": str(temp_output_dir)},
    )

    # (Batch=2, Horizon=12, Dim=1)
    evaluator.update(
        {"predictions": np.random.rand(2, 12, 1), "targets": np.random.rand(2, 12, 1)}
    )
    evaluator.compute(mode="test", epoch=0, step=0)

    file_path = temp_output_dir / "test" / "predictions_epoch_0.nc"
    with xr.open_dataset(file_path) as ds:
        assert "horizon" in ds.dims
        assert "sample" in ds.dims
        assert ds.preds.shape == (2, 12, 1)


def test_reconstruction_netcdf_schema(temp_output_dir):
    """
    Reconstruction Audit: Checks for 'time' and 'feature' consistency.
    """
    evaluator = ReconstructionEvaluator(
        metric_names=["mae"],
        save_predictions=True,
        paths={"eval_details": str(temp_output_dir)},
    )

    # (Batch=2, Time=50, Feat=4)
    evaluator.update(
        {"predictions": np.random.rand(2, 50, 4), "targets": np.random.rand(2, 50, 4)}
    )
    evaluator.compute(mode="train", epoch=1, step=0)

    file_path = temp_output_dir / "train" / "predictions_epoch_1.nc"
    with xr.open_dataset(file_path) as ds:
        assert "sample" in ds.dims
        assert "time" in ds.dims
        assert "feature" in ds.dims
        assert ds.preds.shape == (2, 50, 4)
