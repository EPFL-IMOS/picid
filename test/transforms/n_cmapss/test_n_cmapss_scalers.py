"""Tests for picid.transforms.n_cmapss.n_cmapss_scalers module.

Coverage target: >=95% of picid/transforms/n_cmapss/n_cmapss_scalers.py

Tests verify scaling formulas at known anchor points (mean→0, min→-1, max→+1)
for both numpy and awkward array inputs across standard and min-max modes.
"""

import awkward as ak
import numpy as np
import pytest

from picid.data.data_objects import NamedTransformInput
from picid.transforms.n_cmapss.n_cmapss_scalers import (
    N_CMAPSSDescriptorsScaler,
    N_CMAPSSFeaturesScaler,
    ak_bc,
)

# Hardcoded statistics from the source (repeated here for clarity in test assertions)
W_MEAN = np.array([1.6362398e04, 5.4544175e-01, 6.1369926e01, 4.8835443e02])
W_STD = np.array([8.1254497e03, 1.2108228e-01, 1.8272049e01, 1.9934254e01])
W_MIN = np.array([3.0020000e03, 2.0002499e-01, 2.3730299e01, 4.2319705e02])
W_MAX = np.array([3.5011000e04, 7.3987198e-01, 8.7362656e01, 5.2488281e02])

X_MEAN = np.array(
    [
        5.6809174e02,
        1.3297987e03,
        1.6363848e03,
        1.1242614e03,
        1.2669851e01,
        9.8758097e00,
        1.2862692e01,
        1.5640701e01,
        2.3289339e02,
        2.3705125e02,
        9.8489847e00,
        1.9629447e03,
        8.2361680e03,
        2.4945383e00,
    ]
)
X_STD = np.array(
    [
        2.0833166e01,
        6.7058266e01,
        1.2113522e02,
        6.1629440e01,
        2.8704438e00,
        2.4181337e00,
        2.9141707e00,
        3.4177959e00,
        5.7780758e01,
        5.8610283e01,
        2.7480240e00,
        1.8454973e02,
        2.2242123e02,
        7.6335633e-01,
    ]
)
X_MIN = np.array(
    [
        4.9921100e02,
        1.0889111e03,
        1.2294357e03,
        9.1258221e02,
        6.0976486e00,
        4.4431176e00,
        6.1691909e00,
        7.8722315e00,
        9.1395287e01,
        9.3320122e01,
        4.5537462e00,
        1.4843353e03,
        7.4330903e03,
        7.4892521e-01,
    ]
)
X_MAX = np.array(
    [
        6.3150879e02,
        1.5320127e03,
        1.9813180e03,
        1.3461113e03,
        2.0096666e01,
        1.4716009e01,
        2.0402708e01,
        2.5905502e01,
        4.4721826e02,
        4.5370425e02,
        1.6701591e01,
        2.2791799e03,
        8.8905791e03,
        5.6173835e00,
    ]
)


@pytest.mark.unit
class TestAkBc:
    """Tests for the ak_bc broadcasting helper."""

    def test_broadcasts_1d_to_3d(self):
        """1D numpy array becomes shape (1, 1, n) awkward array.

        **Methodology**: Broadcast [1, 2, 3] and check structure.

        **Expected**: 3D structure with single outer dimensions.
        """
        arr = np.array([1.0, 2.0, 3.0])
        result = ak_bc(arr)
        assert isinstance(result, ak.Array)
        assert result.ndim == 3
        np.testing.assert_array_equal(
            ak.to_numpy(result), arr[np.newaxis, np.newaxis, :]
        )

    def test_preserves_values(self):
        """Broadcasted values are unchanged.

        **Methodology**: Broadcast and compare inner values.

        **Expected**: Innermost values match input.
        """
        arr = np.array([10.0, 20.0])
        result = ak_bc(arr)
        np.testing.assert_array_equal(ak.to_numpy(result)[0, 0], arr)


@pytest.mark.unit
class TestNCMAPSSDescriptorsScaler:
    """Tests for N_CMAPSSDescriptorsScaler (4-feature descriptors)."""

    def test_standard_at_mean_yields_zero(self):
        """Input equal to the hardcoded mean produces all-zero output.

        **Methodology**: Standard scaling: (mean - mean) / std = 0.

        **Expected**: Output ≈ 0 for all features.
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="standard")
        data = NamedTransformInput(descriptors=W_MEAN[np.newaxis, :])
        result = scaler.transform_data(data, {})
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_standard_formula_numpy(self):
        """Standard scaling follows (x - mean) / std.

        **Methodology**: Supply known values, verify formula manually.

        **Expected**: Result matches manual computation.
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="standard")
        raw = np.array([[20000.0, 0.6, 70.0, 500.0]])
        data = NamedTransformInput(descriptors=raw)
        result = scaler.transform_data(data, {})
        expected = (raw - W_MEAN) / W_STD
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_standard_single_sample_shape(self):
        """Single-sample input preserves shape (1, 4).

        **Methodology**: Supply one row.

        **Expected**: Output shape is (1, 4).
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="standard")
        data = NamedTransformInput(descriptors=W_MEAN[np.newaxis, :])
        result = scaler.transform_data(data, {})
        assert result.shape == (1, 4)

    def test_standard_awkward_array(self):
        """Awkward array input triggers ak_bc broadcast path.

        **Methodology**: Supply descriptors as awkward array.

        **Expected**: Result is awkward array with values matching numpy path.
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="standard")
        raw_np = np.array([[20000.0, 0.6, 70.0, 500.0]])
        expected_np = (raw_np - W_MEAN) / W_STD

        raw_ak = ak.Array(raw_np[np.newaxis, :, :])
        data = NamedTransformInput(descriptors=raw_ak)
        result = scaler.transform_data(data, {})
        assert isinstance(result, ak.Array)
        np.testing.assert_allclose(ak.to_numpy(result)[0], expected_np, rtol=1e-6)

    def test_minmax_at_min_yields_minus_one(self):
        """Input equal to min produces output ≈ -1.

        **Methodology**: Min-max: 2*(min-min)/(max-min) - 1 = -1.

        **Expected**: All features ≈ -1.
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="min-max")
        data = NamedTransformInput(descriptors=W_MIN[np.newaxis, :])
        result = scaler.transform_data(data, {})
        np.testing.assert_allclose(result, -1.0, atol=1e-6)

    def test_minmax_at_max_yields_plus_one(self):
        """Input equal to max produces output ≈ +1.

        **Methodology**: Min-max: 2*(max-min)/(max-min) - 1 = +1.

        **Expected**: All features ≈ +1.
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="min-max")
        data = NamedTransformInput(descriptors=W_MAX[np.newaxis, :])
        result = scaler.transform_data(data, {})
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_minmax_midpoint_yields_zero(self):
        """Input at midpoint produces output ≈ 0.

        **Methodology**: midpoint = (min+max)/2 → 2*0.5 - 1 = 0.

        **Expected**: All features ≈ 0.
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="min-max")
        mid = (W_MIN + W_MAX) / 2.0
        data = NamedTransformInput(descriptors=mid[np.newaxis, :])
        result = scaler.transform_data(data, {})
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_minmax_awkward_array(self):
        """Awkward array min-max scaling matches numpy results.

        **Methodology**: Compare awkward and numpy output for same input.

        **Expected**: Values match within tolerance.
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="min-max")

        raw_np = W_MIN[np.newaxis, :]
        data_np = NamedTransformInput(descriptors=raw_np)
        result_np = scaler.transform_data(data_np, {})

        raw_ak = ak.Array(raw_np[np.newaxis, :, :])
        data_ak = NamedTransformInput(descriptors=raw_ak)
        result_ak = scaler.transform_data(data_ak, {})

        assert isinstance(result_ak, ak.Array)
        np.testing.assert_allclose(ak.to_numpy(result_ak)[0], result_np, atol=1e-6)

    def test_output_type_numpy(self):
        """Numpy input returns numpy output.

        **Methodology**: Supply numpy descriptors.

        **Expected**: isinstance(result, np.ndarray).
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="standard")
        data = NamedTransformInput(descriptors=np.random.randn(5, 4))
        result = scaler.transform_data(data, {})
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self):
        """Output shape matches input shape for multi-sample input.

        **Methodology**: Supply (10, 4) input.

        **Expected**: Output is (10, 4).
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="standard")
        raw = np.random.randn(10, 4)
        data = NamedTransformInput(descriptors=raw)
        result = scaler.transform_data(data, {})
        assert result.shape == (10, 4)

    def test_init_stores_scaling(self):
        """Constructor stores scaling mode.

        **Methodology**: Create with scaling="min-max".

        **Expected**: self.scaling == "min-max".
        """
        scaler = N_CMAPSSDescriptorsScaler(scaling="min-max")
        assert scaler.scaling == "min-max"


@pytest.mark.unit
class TestNCMAPSSFeaturesScaler:
    """Tests for N_CMAPSSFeaturesScaler (14-feature sensor data)."""

    def test_standard_at_mean_yields_zero(self):
        """Input equal to feature mean produces all-zero output.

        **Methodology**: Standard scaling at the mean.

        **Expected**: Output ≈ 0 for all 14 features.
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="standard")
        data = NamedTransformInput(features=X_MEAN[np.newaxis, :])
        result = scaler.transform_data(data, {})
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_standard_formula_numpy(self):
        """Standard scaling follows (x - mean) / std.

        **Methodology**: Supply known values, verify against manual formula.

        **Expected**: Result matches (input - X_mean) / X_std.
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="standard")
        raw = X_MEAN + X_STD
        data = NamedTransformInput(features=raw[np.newaxis, :])
        result = scaler.transform_data(data, {})
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_standard_awkward_array(self):
        """Awkward array standard scaling matches numpy.

        **Methodology**: Compare awkward vs numpy for same input.

        **Expected**: Values match.
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="standard")
        raw = X_MEAN[np.newaxis, :]

        data_np = NamedTransformInput(features=raw)
        result_np = scaler.transform_data(data_np, {})

        raw_ak = ak.Array(raw[np.newaxis, :, :])
        data_ak = NamedTransformInput(features=raw_ak)
        result_ak = scaler.transform_data(data_ak, {})

        assert isinstance(result_ak, ak.Array)
        np.testing.assert_allclose(ak.to_numpy(result_ak)[0], result_np, atol=1e-6)

    def test_minmax_at_min_yields_minus_one(self):
        """Min input maps to -1 in min-max scaling.

        **Expected**: All features ≈ -1.
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="min-max")
        data = NamedTransformInput(features=X_MIN[np.newaxis, :])
        result = scaler.transform_data(data, {})
        np.testing.assert_allclose(result, -1.0, atol=1e-6)

    def test_minmax_at_max_yields_plus_one(self):
        """Max input maps to +1 in min-max scaling.

        **Expected**: All features ≈ +1.
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="min-max")
        data = NamedTransformInput(features=X_MAX[np.newaxis, :])
        result = scaler.transform_data(data, {})
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_minmax_midpoint_yields_zero(self):
        """Midpoint maps to 0 in min-max scaling.

        **Expected**: All features ≈ 0.
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="min-max")
        mid = (X_MIN + X_MAX) / 2.0
        data = NamedTransformInput(features=mid[np.newaxis, :])
        result = scaler.transform_data(data, {})
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_minmax_awkward_array(self):
        """Awkward array min-max scaling matches numpy.

        **Methodology**: Compare awkward and numpy for min input.

        **Expected**: Both produce -1.
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="min-max")

        raw_np = X_MIN[np.newaxis, :]
        data_np = NamedTransformInput(features=raw_np)
        result_np = scaler.transform_data(data_np, {})

        raw_ak = ak.Array(raw_np[np.newaxis, :, :])
        data_ak = NamedTransformInput(features=raw_ak)
        result_ak = scaler.transform_data(data_ak, {})

        assert isinstance(result_ak, ak.Array)
        np.testing.assert_allclose(ak.to_numpy(result_ak)[0], result_np, atol=1e-6)

    def test_output_shape_multi_sample(self):
        """Multi-sample input preserves shape.

        **Expected**: (20, 14) → (20, 14).
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="standard")
        raw = np.random.randn(20, 14)
        data = NamedTransformInput(features=raw)
        result = scaler.transform_data(data, {})
        assert result.shape == (20, 14)

    def test_output_type_numpy(self):
        """Numpy input returns numpy output.

        **Expected**: isinstance(result, np.ndarray).
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="standard")
        data = NamedTransformInput(features=np.random.randn(5, 14))
        result = scaler.transform_data(data, {})
        assert isinstance(result, np.ndarray)

    def test_init_stores_scaling(self):
        """Constructor stores scaling mode.

        **Expected**: self.scaling is accessible.
        """
        scaler = N_CMAPSSFeaturesScaler(scaling="min-max")
        assert scaler.scaling == "min-max"
