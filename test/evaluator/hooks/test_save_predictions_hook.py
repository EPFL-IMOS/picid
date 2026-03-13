"""
Tests for picid.evaluator.hooks.save_predictions.SavePredictionsHook.

Ref: docs/evaluators/index.md - DefaultEvaluator, NetCDF saving, dimension naming.
Validates: Guard conditions, _format_xarray branches, dimension conflict resolution.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from picid.evaluator.hooks.save_predictions import SavePredictionsHook


# =============================================================================
# === Tests: Guard Conditions ===
# =============================================================================


def test_save_predictions_hook_returns_early_when_save_disabled():
    """
    Validates guard: hook exits when evaluator.save_predictions is False.

    Methodology: Mock evaluator with save_predictions=False; call on_compute_end.
    Expected outcome: No file I/O; buffer.get_all not required.
    Ref: docs/evaluators/index.md - save_predictions flag.
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    evaluator = MagicMock()
    evaluator.save_predictions = False
    evaluator.buffer = MagicMock()

    hook.on_compute_end({}, evaluator, "val", 0, 0)

    evaluator.buffer.get_all.assert_not_called()


def test_save_predictions_hook_returns_early_when_buffer_empty():
    """
    Validates guard: hook exits when buffer has no data.

    Methodology: buffer.get_all returns {} or preds is None.
    Expected outcome: No _format_xarray, no file write.
    Ref: docs/evaluators/index.md - buffer data flow.
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    evaluator = MagicMock()
    evaluator.save_predictions = True
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {}

    with patch("picid.evaluator.hooks.save_predictions.xr.Dataset") as mock_xr:
        hook.on_compute_end({}, evaluator, "val", 0, 0)
        mock_xr.assert_not_called()


def test_save_predictions_hook_returns_early_when_preds_none():
    """
    Validates guard: hook exits when data has preds=None.

    Methodology: buffer.get_all returns {"targets": ...} without preds.
    Expected outcome: No _format_xarray called (early return).
    Ref: docs/evaluators/index.md - preds/targets contract.
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    evaluator = MagicMock()
    evaluator.save_predictions = True
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {"targets": np.ones((2, 1, 1))}

    with patch("picid.evaluator.hooks.save_predictions.xr.Dataset") as mock_xr:
        hook.on_compute_end({}, evaluator, "val", 0, 0)
        mock_xr.assert_not_called()


def test_save_predictions_hook_reraises_value_error_from_format_xarray():
    """
    Validates error path: _format_xarray ValueError is logged and re-raised.

    Methodology: Hook with dims=None; buffer has valid data; _format_xarray raises.
    Expected outcome: ValueError propagated after logging.
    Ref: docs/evaluators/index.md - dimension schema validation.
    """
    hook = SavePredictionsHook(dims=None)
    evaluator = MagicMock()
    evaluator.save_predictions = True
    evaluator.paths = MagicMock(eval_details="/mock")
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {
        "preds": np.ones((2, 1, 1)),
        "targets": np.ones((2, 1, 1)),
    }

    with pytest.raises(ValueError, match="No dimension names provided"):
        hook.on_compute_end({}, evaluator, "val", 0, 0)


def test_save_predictions_hook_returns_early_when_eval_details_missing():
    """
    Validates guard: hook exits when evaluator.paths.eval_details is None.

    Methodology: paths.eval_details = None; _format_xarray succeeds.
    Expected outcome: No file write; logger.warning issued.
    Ref: docs/evaluators/index.md - paths.eval_details for output dir.
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    evaluator = MagicMock()
    evaluator.save_predictions = True
    evaluator.paths = MagicMock()
    evaluator.paths.eval_details = None
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {
        "preds": np.ones((2, 1, 1)),
        "targets": np.ones((2, 1, 1)),
    }

    with patch("picid.evaluator.hooks.save_predictions.xr.Dataset") as mock_xr:
        hook.on_compute_end({}, evaluator, "val", 0, 0)
        mock_xr.assert_not_called()


# =============================================================================
# === Tests: _format_xarray Validation and Branches ===
# =============================================================================


def test_format_xarray_raises_when_dims_none():
    """
    Validates error path: _format_xarray raises when dims is None.

    Methodology: SavePredictionsHook(dims=None); call _format_xarray.
    Expected outcome: ValueError with message about initialization.
    Ref: docs/evaluators/index.md - dimension naming schema.
    """
    hook = SavePredictionsHook(dims=None)
    data = {"preds": np.ones((2, 1, 1)), "targets": np.ones((2, 1, 1))}
    evaluator = MagicMock()

    with pytest.raises(ValueError, match="No dimension names provided"):
        hook._format_xarray(data, evaluator)


def test_format_xarray_raises_when_dims_not_three_elements():
    """
    Validates error path: _format_xarray raises when len(dims) != 3.

    Methodology: SavePredictionsHook(dims=["a", "b"]); call _format_xarray.
    Expected outcome: ValueError with message about 3 elements.
    Ref: docs/evaluators/index.md - 3D tensors (N, T, C).
    """
    hook = SavePredictionsHook(dims=["sample", "time"])
    data = {"preds": np.ones((2, 1, 1)), "targets": np.ones((2, 1, 1))}
    evaluator = MagicMock()

    with pytest.raises(ValueError, match="exactly 3 elements"):
        hook._format_xarray(data, evaluator)


def test_format_xarray_regression_data_preserves_shapes(phm_regression_buffer_data):
    """
    Validates regression branch: preds/targets with same feature dim.

    Methodology: phm_regression_buffer_data (N, 1, 1); both same shape.
    Gold Standard: No dimension conflict; standard dims applied.
    Expected outcome: preds and targets with dims ["sample","time","feature"].
    Ref: docs/evaluators/index.md - regression shape (N, T, 1).
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    data = phm_regression_buffer_data
    evaluator = MagicMock()

    result = hook._format_xarray(data, evaluator)

    assert "preds" in result
    assert "targets" in result
    assert result["preds"][0] == ["sample", "time", "feature"]
    assert result["targets"][0] == ["sample", "time", "feature"]
    assert result["preds"][1].shape == data["preds"].shape
    assert result["targets"][1].shape == data["targets"].shape


def test_format_xarray_classification_resolves_dimension_conflict(
    phm_classification_buffer_data,
):
    """
    Validates classification branch: preds (N,T,C) vs targets (N,T,1) → _label rename.

    Methodology: phm_classification_buffer_data - preds (8,1,5), targets (8,1,1).
    Gold Standard: Conflict resolution; targets get feature_label dim.
    Expected outcome: preds use "feature", targets use "feature_label".
    Ref: docs/evaluators/index.md - ClassificationEvaluator, logits vs labels.
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    data = phm_classification_buffer_data
    evaluator = MagicMock()

    result = hook._format_xarray(data, evaluator)

    assert result["preds"][0] == ["sample", "time", "feature"]
    assert result["targets"][0] == ["sample", "time", "feature_label"]
    assert result["preds"][1].shape == data["preds"].shape
    assert result["targets"][1].shape == data["targets"].shape


def test_format_xarray_includes_norm_preds_and_norm_targets():
    """
    Validates dual reporting: norm_preds and norm_targets mapped to _normalized.

    Methodology: Buffer with preds, targets, norm_preds, norm_targets.
    Expected outcome: preds_normalized, targets_normalized in output.
    Ref: docs/evaluators/index.md - dual reporting normalized/denormalized.
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    data = {
        "preds": np.ones((2, 1, 1)),
        "targets": np.ones((2, 1, 1)),
        "norm_preds": np.zeros((2, 1, 1)),
        "norm_targets": np.zeros((2, 1, 1)),
    }
    evaluator = MagicMock()

    result = hook._format_xarray(data, evaluator)

    assert "preds_normalized" in result
    assert "targets_normalized" in result
    assert result["preds_normalized"][0] == ["sample", "time", "feature"]
    assert result["targets_normalized"][0] == ["sample", "time", "feature"]
    np.testing.assert_array_equal(result["preds_normalized"][1], data["norm_preds"])
    np.testing.assert_array_equal(result["targets_normalized"][1], data["norm_targets"])


def test_format_xarray_skips_none_keys():
    """
    Validates that None values for optional keys are skipped.

    Methodology: Only preds and targets; norm_preds, norm_targets absent.
    Expected outcome: Only preds and targets in result.
    Ref: docs/evaluators/index.md - optional norm_* for dual reporting.
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    data = {"preds": np.ones((2, 1, 1)), "targets": np.ones((2, 1, 1))}
    evaluator = MagicMock()

    result = hook._format_xarray(data, evaluator)

    assert "preds" in result
    assert "targets" in result
    assert "preds_normalized" not in result
    assert "targets_normalized" not in result


def test_format_xarray_adds_unit_ids(phm_multiunit_buffer_data):
    """
    Validates unit_ids: added with sample + unit_dim_* dims.

    Methodology: phm_multiunit_buffer_data includes unit_ids (15,).
    Expected outcome: unit_ids in result with dims [sample, unit_dim_1] etc.
    Ref: docs/evaluators/index.md - MultiUnitEvaluator, unit_id in model_out.
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    data = phm_multiunit_buffer_data
    evaluator = MagicMock()

    result = hook._format_xarray(data, evaluator)

    assert "unit_ids" in result
    u_dims, u_data = result["unit_ids"]
    assert "sample" in u_dims
    assert u_data.shape[0] == data["preds"].shape[0]


def test_format_xarray_non_3d_uses_fallback_dims():
    """
    Validates fallback: non-3D arrays get dim_0, dim_1, ... .

    Methodology: 2D array (e.g. flattened preds) in buffer.
    Expected outcome: fallback_dims = [f"dim_{i}" for i in range(ndim)].
    Ref: docs/evaluators/index.md - fallback for non-standard shapes.
    """
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    data = {"preds": np.ones((10, 5)), "targets": np.ones((10, 5))}
    evaluator = MagicMock()

    result = hook._format_xarray(data, evaluator)

    assert result["preds"][0] == ["dim_0", "dim_1"]
    assert result["targets"][0] == ["dim_0", "dim_1"]


# =============================================================================
# === Tests: Full on_compute_end Flow ===
# =============================================================================


@patch("picid.evaluator.hooks.save_predictions.xr.Dataset")
@patch("picid.evaluator.hooks.save_predictions.Path")
def test_save_predictions_hook_writes_netcdf(mock_path_cls, mock_xr_dataset):
    """
    Validates full flow: on_compute_end formats data and writes NetCDF.

    Methodology: Mock Path and xr.Dataset; call on_compute_end with valid data.
    Expected outcome: mkdir called, Dataset created, to_netcdf called.
    Ref: docs/evaluators/index.md - predictions saved to netCDF.
    """
    mock_out_path = MagicMock()
    mock_path = MagicMock()
    mock_path.__truediv__ = MagicMock(return_value=mock_out_path)
    mock_path_cls.return_value = mock_path
    mock_ds = MagicMock()
    mock_xr_dataset.return_value = mock_ds

    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    evaluator = MagicMock()
    evaluator.save_predictions = True
    evaluator.paths = MagicMock()
    evaluator.paths.eval_details = "/mock/eval"
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {
        "preds": np.ones((3, 1, 1)),
        "targets": np.ones((3, 1, 1)),
    }

    hook.on_compute_end({}, evaluator, "val", 1, 10)

    mock_out_path.mkdir.assert_called_with(parents=True, exist_ok=True)
    mock_xr_dataset.assert_called_once()
    mock_ds.to_netcdf.assert_called_once()
