"""
Tests for picid.evaluator.scaling_wrapper module.

This module contains tests for ScalingWrapper and MultivariateTimeseriesScalingWrapper,
which handle inverse scaling operations critical for converting normalized
PHM predictions back to physical units (e.g., cycles, hours).

Coverage Target: 100%

PHM Context:
- RUL predictions are typically normalized to [0, 1] during training
- Inverse scaling converts predictions back to actual remaining cycles
- Multivariate scaling handles multiple sensor channels simultaneously
"""

import pytest
import torch
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from picid.evaluator.scaling_wrapper import (
    ScalingWrapper,
    MultivariateTimeseriesScalingWrapper,
)
from picid.data.data_objects import NamedTransformInput
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# === Inverse adapters (sklearn + simple mocks) ===
# =============================================================================


class MinMaxScalerNamedAdapter:
    """
    Wraps sklearn MinMaxScaler with the NamedTransformInput contract used by
    :class:`~picid.evaluator.scaling_wrapper.ScalingWrapper`.
    """

    def __init__(self, feature_min: float = 0.0, feature_max: float = 10.0):
        self._scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._scaler.fit(np.array([[feature_min], [feature_max]], dtype=np.float64))

    def inverse_transform(self, data: NamedTransformInput, metadata=None):
        if getattr(data, "predictions", None) is not None:
            arr = np.asarray(data.predictions, dtype=np.float64)
        else:
            arr = np.asarray(data.targets, dtype=np.float64)
        orig_shape = arr.shape
        flat = arr.reshape(-1, orig_shape[-1])
        out = self._scaler.inverse_transform(flat).astype(arr.dtype)
        return out.reshape(orig_shape)


# =============================================================================
# === Mock Inverse Transform (simple closed form) ===
# =============================================================================


class MockInverseTransform:
    """
    Mock inverse transform for testing scaling wrapper.

    Simulates a scaler that multiplies values by 100 (e.g., converting
    normalized RUL [0, 1] to actual cycles [0, 100]).
    """

    def inverse_transform(self, data: NamedTransformInput, metadata=None):
        """Apply inverse transform (multiply by 100)."""
        # Extract the array from NamedTransformInput
        if hasattr(data, "predictions"):
            arr = data.predictions
        elif hasattr(data, "targets"):
            arr = data.targets
        else:
            # Fallback for dictionary-like access
            for key in ["predictions", "targets"]:
                if key in data.__dict__:
                    arr = getattr(data, key)
                    break

        return arr * 100.0


class MockInverseTransformWithMetadata:
    """
    Mock inverse transform that uses metadata (e.g., per-unit scaling).

    PHM Context: Different equipment units may have different scaling
    factors based on their operational history.
    """

    def inverse_transform(self, data: NamedTransformInput, metadata=None):
        """Apply inverse transform with per-unit scaling."""
        # Extract the array from NamedTransformInput
        if hasattr(data, "predictions"):
            arr = data.predictions
        elif hasattr(data, "targets"):
            arr = data.targets
        else:
            for key in ["predictions", "targets"]:
                if key in data.__dict__:
                    arr = getattr(data, key)
                    break

        # Apply unit-specific scaling if metadata provided
        if metadata and "unit_id" in metadata:
            unit_id = metadata["unit_id"]
            # Unit 1 scales by 50, Unit 2 scales by 200
            scale_factor = 50.0 if unit_id == 1 else 200.0
            return arr * scale_factor

        return arr * 100.0


# =============================================================================
# === Test Class for ScalingWrapper ===
# =============================================================================


class TestScalingWrapper:
    """Tests for ScalingWrapper class.

    ScalingWrapper handles inverse scaling for univariate and regression tasks
    where predictions have shape (B, 1, 1) or (B, T, 1).
    """

    def test_init_basic(self):
        """Test basic initialization of ScalingWrapper.

        **PHM Logic**: ScalingWrapper stores scaler reference and flags.

        **Methodology**: Create wrapper and verify attributes.

        **Expected**: Attributes correctly set.

        Validates: Requirement SW-INIT-1 - Basic initialization
        """
        mock_transform = MockInverseTransform()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="regression",
        )

        assert wrapper.inverse_transform == mock_transform
        assert wrapper.apply_inverse is True
        assert wrapper.task_mode == "regression"

    def test_init_without_scaler(self):
        """Test initialization without scaler (no inverse transform).

        **PHM Logic**: Some tasks don't require inverse scaling.

        **Methodology**: Create wrapper with None scaler.

        **Expected**: inverse_transform is None.

        Validates: Requirement SW-INIT-2 - No scaler initialization
        """
        wrapper = ScalingWrapper(
            inverse_transform=None,
            apply_inverse_scaling=True,
        )

        assert wrapper.inverse_transform is None
        assert wrapper.apply_inverse is True

    def test_inverse_transform_if_needed_when_disabled(self):
        """Test inverse_transform_if_needed returns unchanged when disabled.

        **PHM Logic**: When apply_inverse=False, data passes through unchanged.

        **Methodology**: Call with apply_inverse=False.

        **Expected**: Input returned unchanged.

        Validates: Requirement SW-INV-1 - Disabled passthrough
        """
        mock_transform = MockInverseTransform()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=False,
        )

        preds = np.array([[[0.5]], [[0.7]]])  # Shape (2, 1, 1)
        targets = np.array([[[0.4]], [[0.6]]])

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        assert_array_equal(result_preds, preds)
        assert_array_equal(result_targets, targets)

    def test_inverse_transform_if_needed_no_scaler(self):
        """Test inverse_transform_if_needed with no scaler.

        **PHM Logic**: Without scaler, data passes through unchanged.

        **Methodology**: Call with inverse_transform=None.

        **Expected**: Input returned unchanged.

        Validates: Requirement SW-INV-2 - No scaler passthrough
        """
        wrapper = ScalingWrapper(
            inverse_transform=None,
            apply_inverse_scaling=True,
        )

        preds = np.array([[[0.5]]])
        targets = np.array([[[0.4]]])

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        assert_array_equal(result_preds, preds)
        assert_array_equal(result_targets, targets)

    def test_inverse_transform_3d_regression(self):
        """Test inverse transform for 3D regression data (B, T=1, 1).

        **PHM Logic**: RUL predictions are typically (batch, 1, 1) shaped.
        Inverse scaling converts normalized [0,1] to actual cycles.

        **Methodology**: Use mock scaler (x100) on 3D data.

        **Expected**: Values multiplied by 100, shape preserved.

        Validates: Requirement SW-INV-3 - 3D regression inverse
        """
        mock_transform = MockInverseTransform()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="regression",
        )

        # Normalized RUL: 0.5 = 50% remaining life
        preds = np.array([[[0.5]], [[0.7]]])  # Shape (2, 1, 1)
        targets = np.array([[[0.4]], [[0.6]]])

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        # Scaled: 0.5 * 100 = 50 cycles
        expected_preds = np.array([[[50.0]], [[70.0]]])
        expected_targets = np.array([[[40.0]], [[60.0]]])

        assert_array_almost_equal(result_preds, expected_preds, decimal=5)
        assert_array_almost_equal(result_targets, expected_targets, decimal=5)

    def test_inverse_transform_3d_forecasting(self):
        """Test inverse transform for 3D forecasting data (B, T>1, 1).

        **PHM Logic**: Forecasting predictions have multiple timesteps.
        Each timestep should be independently inverse transformed.

        **Methodology**: Use mock scaler on multi-timestep data.

        **Expected**: All timesteps transformed, shape preserved.

        Validates: Requirement SW-INV-4 - 3D forecasting inverse
        """
        mock_transform = MockInverseTransform()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="forecasting",
        )

        # Shape (2, 5, 1) - 2 samples, 5 timesteps
        preds = np.array(
            [
                [[0.1], [0.2], [0.3], [0.4], [0.5]],
                [[0.6], [0.7], [0.8], [0.9], [1.0]],
            ]
        )
        targets = preds.copy()

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        # All values scaled by 100
        expected = preds * 100.0
        assert_array_almost_equal(result_preds, expected, decimal=5)

    def test_inverse_transform_2d_data(self):
        """Test inverse transform for 2D data (B, C).

        **PHM Logic**: Some legacy formats use 2D arrays directly.

        **Methodology**: Use mock scaler on 2D data.

        **Expected**: Values transformed correctly.

        Validates: Requirement SW-INV-5 - 2D data inverse
        """
        mock_transform = MockInverseTransform()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
        )

        preds = np.array([[0.5], [0.7]])  # Shape (2, 1)
        targets = np.array([[0.4], [0.6]])

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        expected_preds = np.array([[50.0], [70.0]])
        assert_array_almost_equal(result_preds, expected_preds, decimal=5)

    def test_inverse_transform_torch_tensor(self):
        """Test inverse transform converts torch tensors to numpy.

        **PHM Logic**: Model outputs are often PyTorch tensors that
        need conversion for metric computation.

        **Methodology**: Pass torch tensors, verify numpy output.

        **Expected**: Tensors converted to numpy and transformed.

        Validates: Requirement SW-INV-6 - Tensor conversion
        """
        mock_transform = MockInverseTransform()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="regression",
        )

        preds = torch.tensor([[[0.5]], [[0.7]]])  # Shape (2, 1, 1)
        targets = torch.tensor([[[0.4]], [[0.6]]])

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        # Should be numpy arrays after conversion
        assert isinstance(result_preds, np.ndarray)
        assert isinstance(result_targets, np.ndarray)

        expected_preds = np.array([[[50.0]], [[70.0]]])
        assert_array_almost_equal(result_preds, expected_preds, decimal=5)

    def test_inverse_transform_with_metadata(self):
        """Test inverse transform uses metadata for unit-specific scaling.

        **PHM Logic**: Different equipment units may have different
        normalization factors based on operational history.

        **Methodology**: Pass metadata with unit_id.

        **Expected**: Unit-specific scaling applied.

        Validates: Requirement SW-INV-7 - Metadata-based scaling
        """
        mock_transform = MockInverseTransformWithMetadata()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="regression",
        )

        preds = np.array([[[1.0]]])  # Shape (1, 1, 1)
        targets = np.array([[[1.0]]])

        # Unit 1: scale by 50
        result_preds_u1, _ = wrapper.inverse_transform_if_needed(
            preds, targets, metadata={"unit_id": 1}
        )
        assert_array_almost_equal(result_preds_u1, np.array([[[50.0]]]), decimal=5)

        # Unit 2: scale by 200
        result_preds_u2, _ = wrapper.inverse_transform_if_needed(
            preds, targets, metadata={"unit_id": 2}
        )
        assert_array_almost_equal(result_preds_u2, np.array([[[200.0]]]), decimal=5)

    def test_inverse_transform_real_sklearn_minmax_matches_closed_form(self):
        """Inverse path through sklearn scaler matches manual (feature_range 0–1, train 0–10)."""
        adapter = MinMaxScalerNamedAdapter(feature_min=0.0, feature_max=10.0)
        wrapper = ScalingWrapper(
            inverse_transform=adapter,
            apply_inverse_scaling=True,
            task_mode="regression",
        )
        preds = np.array([[[0.5]], [[0.1]]], dtype=np.float64)
        targets = np.array([[[0.3]], [[0.9]]], dtype=np.float64)
        got_p, got_t = wrapper.inverse_transform_if_needed(preds, targets)
        # 0.5 in [0,1] maps back to 5.0 on [0,10]; 0.1 -> 1.0, etc.
        assert_array_almost_equal(got_p, np.array([[[5.0]], [[1.0]]]), decimal=5)
        assert_array_almost_equal(got_t, np.array([[[3.0]], [[9.0]]]), decimal=5)

    def test_inverse_transform_invalid_shape(self):
        """Test inverse transform raises error for invalid shapes.

        **PHM Logic**: Only 2D and 3D data are supported.

        **Methodology**: Pass 1D data.

        **Expected**: ValueError raised.

        Validates: Requirement SW-INV-8 - Shape validation
        """
        mock_transform = MockInverseTransform()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
        )

        # 1D array - invalid
        preds = np.array([0.5, 0.7])
        targets = np.array([0.4, 0.6])

        with pytest.raises(ValueError, match="Unexpected shape"):
            wrapper.inverse_transform_if_needed(preds, targets)

    def test_inverse_transform_non_forecasting_time_dim_assertion(self):
        """Test assertion for non-forecasting mode with time dim != 1.

        **PHM Logic**: In regression mode, time dimension should be 1.

        **Methodology**: Pass 3D data with T>1 in non-forecasting mode.

        **Expected**: AssertionError raised.

        Validates: Requirement SW-INV-9 - Time dimension validation
        """
        mock_transform = MockInverseTransform()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="regression",  # Not forecasting
        )

        # Shape (2, 5, 1) - T=5 invalid for non-forecasting
        preds = np.array(
            [
                [[0.1], [0.2], [0.3], [0.4], [0.5]],
                [[0.6], [0.7], [0.8], [0.9], [1.0]],
            ]
        )
        targets = preds.copy()

        with pytest.raises(AssertionError):
            wrapper.inverse_transform_if_needed(preds, targets)

    def test_inverse_transform_forecasting_feature_dim_assertion(self):
        """Test assertion for forecasting mode with feature dim != 1.

        **PHM Logic**: Currently only univariate forecasting is supported.

        **Methodology**: Pass 3D data with C>1 in forecasting mode.

        **Expected**: AssertionError raised.

        Validates: Requirement SW-INV-10 - Feature dimension validation
        """
        mock_transform = MockInverseTransform()
        wrapper = ScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="forecasting",
        )

        # Shape (2, 5, 3) - C=3 invalid for univariate forecasting
        preds = np.ones((2, 5, 3))
        targets = preds.copy()

        with pytest.raises(AssertionError):
            wrapper.inverse_transform_if_needed(preds, targets)


# =============================================================================
# === Test Class for MultivariateTimeseriesScalingWrapper ===
# =============================================================================


class TestMultivariateTimeseriesScalingWrapper:
    """Tests for MultivariateTimeseriesScalingWrapper class.

    This wrapper handles inverse scaling for multivariate time series
    where predictions have shape (B, T, C) with C > 1.
    """

    def test_init_multivariate_mode_assertion(self):
        """Test initialization requires multivariate task_mode.

        **PHM Logic**: This wrapper is specifically for multivariate data.

        **Methodology**: Try to create with non-multivariate mode.

        **Expected**: AssertionError raised.

        Validates: Requirement MSW-INIT-1 - Mode assertion
        """
        with pytest.raises(AssertionError, match="multivariate"):
            MultivariateTimeseriesScalingWrapper(
                inverse_transform=MockInverseTransform(),
                apply_inverse_scaling=True,
                task_mode="regression",  # Not multivariate
            )

    def test_init_multivariate_mode_success(self):
        """Test successful initialization with multivariate mode.

        **PHM Logic**: Wrapper accepts multivariate task mode.

        **Methodology**: Create with task_mode="multivariate".

        **Expected**: Wrapper created successfully.

        Validates: Requirement MSW-INIT-2 - Multivariate initialization
        """
        wrapper = MultivariateTimeseriesScalingWrapper(
            inverse_transform=MockInverseTransform(),
            apply_inverse_scaling=True,
            task_mode="multivariate",
        )

        assert wrapper.task_mode == "multivariate"

    def test_inverse_transform_when_disabled(self):
        """Test inverse_transform_if_needed returns unchanged when disabled.

        **PHM Logic**: When apply_inverse=False, data passes through.

        **Methodology**: Call with apply_inverse=False.

        **Expected**: Input returned unchanged.

        Validates: Requirement MSW-INV-1 - Disabled passthrough
        """
        wrapper = MultivariateTimeseriesScalingWrapper(
            inverse_transform=MockInverseTransform(),
            apply_inverse_scaling=False,
            task_mode="multivariate",
        )

        preds = np.ones((2, 10, 4))  # (batch, time, features)
        targets = preds.copy()

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        assert_array_equal(result_preds, preds)
        assert_array_equal(result_targets, targets)

    def test_inverse_transform_no_scaler(self):
        """Test inverse_transform_if_needed with no scaler.

        **PHM Logic**: Without scaler, data passes through unchanged.

        **Methodology**: Create with inverse_transform=None.

        **Expected**: Input returned unchanged.

        Validates: Requirement MSW-INV-2 - No scaler passthrough
        """
        wrapper = MultivariateTimeseriesScalingWrapper(
            inverse_transform=None,
            apply_inverse_scaling=True,
            task_mode="multivariate",
        )

        preds = np.ones((2, 10, 4))
        targets = preds.copy()

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        assert_array_equal(result_preds, preds)
        assert_array_equal(result_targets, targets)

    def test_inverse_transform_3d_multivariate(self):
        """Test inverse transform for 3D multivariate data (B, T, C).

        **PHM Logic**: Multivariate sensor data with multiple channels
        (e.g., temperature, pressure, vibration) at each timestep.

        **Methodology**: Use mock scaler on multivariate data.

        **Expected**: All channels transformed, shape preserved.

        Validates: Requirement MSW-INV-3 - 3D multivariate inverse
        """
        mock_transform = MockInverseTransform()
        wrapper = MultivariateTimeseriesScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="multivariate",
        )

        # Shape (2, 5, 3) - 2 samples, 5 timesteps, 3 sensors
        preds = np.array(
            [
                [
                    [0.1, 0.2, 0.3],
                    [0.2, 0.3, 0.4],
                    [0.3, 0.4, 0.5],
                    [0.4, 0.5, 0.6],
                    [0.5, 0.6, 0.7],
                ],
                [
                    [0.6, 0.7, 0.8],
                    [0.7, 0.8, 0.9],
                    [0.8, 0.9, 1.0],
                    [0.9, 1.0, 0.1],
                    [1.0, 0.1, 0.2],
                ],
            ]
        )
        targets = preds.copy()

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        # All values scaled by 100
        expected = preds * 100.0
        assert_array_almost_equal(result_preds, expected, decimal=5)
        assert result_preds.shape == preds.shape

    def test_inverse_transform_torch_tensor(self):
        """Test inverse transform converts torch tensors to numpy.

        **PHM Logic**: Model outputs are often PyTorch tensors.

        **Methodology**: Pass torch tensors, verify numpy output.

        **Expected**: Tensors converted to numpy and transformed.

        Validates: Requirement MSW-INV-4 - Tensor conversion
        """
        mock_transform = MockInverseTransform()
        wrapper = MultivariateTimeseriesScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="multivariate",
        )

        preds = torch.randn(2, 5, 3)  # Shape (2, 5, 3)
        targets = torch.randn(2, 5, 3)

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        assert isinstance(result_preds, np.ndarray)
        assert isinstance(result_targets, np.ndarray)

    def test_inverse_transform_invalid_shape(self):
        """Test inverse transform raises error for non-3D shapes.

        **PHM Logic**: Multivariate wrapper requires 3D data.

        **Methodology**: Pass 2D data.

        **Expected**: ValueError raised.

        Validates: Requirement MSW-INV-5 - Shape validation
        """
        mock_transform = MockInverseTransform()
        wrapper = MultivariateTimeseriesScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="multivariate",
        )

        # 2D array - invalid for multivariate
        preds = np.array([[0.5, 0.7], [0.4, 0.6]])
        targets = preds.copy()

        with pytest.raises(ValueError, match="need to be 3D"):
            wrapper.inverse_transform_if_needed(preds, targets)

    def test_inverse_transform_batch_size_one(self):
        """Test inverse transform with batch size 1.

        **PHM Logic**: Single-sample batches should work correctly.

        **Methodology**: Pass data with batch_size=1.

        **Expected**: Correct transformation and shape.

        Validates: Requirement MSW-INV-6 - Batch size one handling
        """
        mock_transform = MockInverseTransform()
        wrapper = MultivariateTimeseriesScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="multivariate",
        )

        # Shape (1, 10, 4) - single sample
        preds = np.ones((1, 10, 4)) * 0.5
        targets = preds.copy()

        result_preds, result_targets = wrapper.inverse_transform_if_needed(
            preds, targets
        )

        expected = np.ones((1, 10, 4)) * 50.0
        assert_array_almost_equal(result_preds, expected, decimal=5)
        assert result_preds.shape == (1, 10, 4)

    def test_inverse_transform_with_metadata(self):
        """Metadata (e.g. unit_id) is forwarded like :class:`ScalingWrapper`."""
        mock_transform = MockInverseTransformWithMetadata()
        wrapper = MultivariateTimeseriesScalingWrapper(
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="multivariate",
        )

        preds = np.ones((1, 2, 3))
        targets = preds.copy()

        result_preds, _ = wrapper.inverse_transform_if_needed(
            preds, targets, metadata={"unit_id": 1}
        )
        assert_array_almost_equal(result_preds, np.ones((1, 2, 3)) * 50.0, decimal=5)
