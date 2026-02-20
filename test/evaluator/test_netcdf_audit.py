import pytest
import xarray as xr
import numpy as np
from types import SimpleNamespace
from picid.evaluator.default import DefaultEvaluator
from picid.evaluator.classification import ClassificationEvaluator
from picid.evaluator.forecasting import ForecastingEvaluator
from picid.evaluator.reconstruction import ReconstructionEvaluator
from picid.evaluator.hooks.save_predictions import SavePredictionsHook


@pytest.fixture
def temp_output_dir(tmp_path):
    """Creates a temporary root for artifact validation."""
    return tmp_path / "outputs"


@pytest.fixture
def mock_paths(temp_output_dir):
    """Wraps the path in a namespace to satisfy getattr(self.paths, 'eval_details')."""
    return SimpleNamespace(eval_details=str(temp_output_dir))


@pytest.fixture
def standard_dims():
    """Returns the standardized dimension names for PHM tasks."""
    return ["sample", "time", "feature"]


def test_classification_netcdf_schema(temp_output_dir, mock_paths, standard_dims):
    num_classes = 5
    evaluator = ClassificationEvaluator(
        metric_names=["accuracy"],
        num_classes=num_classes,
        save_predictions=True,
        paths=mock_paths,
    )
    evaluator.add_hook(SavePredictionsHook(dims=standard_dims))

    evaluator.update(
        {
            "predictions": np.random.rand(4, 1, num_classes),
            "targets": np.random.randint(0, num_classes, (4, 1, 1)),
        }
    )

    evaluator.compute(mode="test", epoch=5, step=100)

    file_path = temp_output_dir / "test" / "predictions.nc"
    with xr.open_dataset(file_path) as ds:
        assert "sample" in ds.dims
        assert "time" in ds.dims
        # 'feature' comes from preds (size 5)
        assert ds.dims["feature"] == num_classes
        # 'feature_label' comes from targets (size 1)
        assert ds.dims["feature_label"] == 1

        assert ds["preds"].shape == (4, 1, 5)
        assert ds["targets"].shape == (4, 1, 1)


def test_default_regression_netcdf_schema(temp_output_dir, mock_paths, standard_dims):
    """Audit: Standard regression (N, T, 1) check."""
    evaluator = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        save_predictions=True,
        paths=mock_paths,
    )

    # FIX: Initialize with mandatory dimensions
    evaluator.add_hook(SavePredictionsHook(dims=standard_dims))

    # STRICT 3D: (N, T, 1)
    evaluator.update(
        {"predictions": np.random.rand(5, 1, 1), "targets": np.random.rand(5, 1, 1)}
    )
    evaluator.compute(mode="val", epoch=2, step=0)

    file_path = temp_output_dir / "val" / "predictions.nc"
    assert file_path.exists()

    with xr.open_dataset(file_path) as ds:
        assert ds["preds"].shape == (5, 1, 1)
        assert ds["targets"].shape == (5, 1, 1)
        assert "time" in ds.dims


def test_forecasting_netcdf_schema(temp_output_dir, mock_paths, standard_dims):
    """Audit: Checks forecasting Horizon/Time dimension mapping."""
    evaluator = ForecastingEvaluator(
        target_dim_position=0,
        metric_names=["mse"],
        model_seq_len=96,
        model_label_len=48,
        model_pred_len=24,
        effective_pred_len=12,
        save_predictions=True,
        paths=mock_paths,
    )

    # FIX: Initialize with mandatory dimensions
    evaluator.add_hook(SavePredictionsHook(dims=standard_dims))

    # STRICT 3D: (Batch=2, Horizon=24, Feature=1)
    evaluator.update(
        {"predictions": np.random.rand(2, 24, 1), "targets": np.random.rand(2, 24, 1)}
    )
    evaluator.compute(mode="test", epoch=0, step=0)

    file_path = temp_output_dir / "test" / "predictions.nc"
    assert file_path.exists()

    with xr.open_dataset(file_path) as ds:
        # Expecting (Batch, Time, Feature) -> (2, 12, 1)
        assert ds["preds"].shape == (2, 12, 1)
        assert "time" in ds.dims


def test_reconstruction_netcdf_schema(temp_output_dir, mock_paths, standard_dims):
    """Audit: Ensures T and D consistency for reconstruction tasks."""
    evaluator = ReconstructionEvaluator(
        metric_names=["mae"], save_predictions=True, paths=mock_paths
    )

    # FIX: Initialize with mandatory dimensions
    evaluator.add_hook(SavePredictionsHook(dims=standard_dims))

    # STRICT 3D: (Batch=2, Time=50, Feature=1)
    evaluator.update(
        {"predictions": np.random.rand(2, 50, 1), "targets": np.random.rand(2, 50, 1)}
    )
    evaluator.compute(mode="train", epoch=1, step=0)

    file_path = temp_output_dir / "train" / "predictions.nc"
    assert file_path.exists()

    with xr.open_dataset(file_path) as ds:
        assert ds["preds"].shape == (2, 50, 1)
        assert "time" in ds.dims
        assert "feature" in ds.dims
