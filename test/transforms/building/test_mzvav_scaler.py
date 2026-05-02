"""Tests for picid.transforms.building.mzvav_scaler."""

from __future__ import annotations

import numpy as np
import awkward as ak
import pytest

from picid.transforms.building.mzvav_scaler import MinMaxScalerMZVAV


def _make_data(values: np.ndarray) -> dict:
    """Wrap a 3-D numpy array as the single-key NamedTransformInput expected by the scaler."""
    return {"features": ak.from_numpy(values)}


@pytest.mark.unit
class TestMinMaxScalerMZVAV:
    """Tests for MinMaxScalerMZVAV fit, transform, and inverse_transform."""

    def _scaler_and_data(self):
        scaler = MinMaxScalerMZVAV()
        # shape (2, 3, 4): 2 segments, 3 timesteps, 4 features
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        data = _make_data(values)
        return scaler, data, values

    def test_fit_data_does_not_raise(self):
        scaler, data, _ = self._scaler_and_data()
        scaler.fit_data(data, metadata={})

    def test_transform_data_output_is_awkward(self):
        scaler, data, _ = self._scaler_and_data()
        scaler.fit_data(data, metadata={})
        result = scaler.transform_data(data, metadata={})
        assert isinstance(result, ak.Array)

    def test_transform_data_scales_to_unit_range(self):
        scaler, data, _ = self._scaler_and_data()
        scaler.fit_data(data, metadata={})
        result = scaler.transform_data(data, metadata={})
        arr = ak.to_numpy(ak.to_regular(result))
        assert arr.min() >= 0.0 - 1e-6
        assert arr.max() <= 1.0 + 1e-6

    def test_transform_data_preserves_shape(self):
        scaler, data, values = self._scaler_and_data()
        scaler.fit_data(data, metadata={})
        result = scaler.transform_data(data, metadata={})
        arr = ak.to_numpy(ak.to_regular(result))
        assert arr.shape == values.shape

    def test_inverse_transform_recovers_original(self):
        scaler, data, values = self._scaler_and_data()
        scaler.fit_data(data, metadata={})
        transformed = scaler.transform_data(data, metadata={})
        transformed_data = {"features": transformed}
        recovered = scaler.inverse_transform(transformed_data, metadata={})
        arr = ak.to_numpy(ak.to_regular(recovered))
        np.testing.assert_allclose(arr, values, atol=1e-4)

    def test_inverse_transform_output_is_awkward(self):
        scaler, data, _ = self._scaler_and_data()
        scaler.fit_data(data, metadata={})
        transformed = scaler.transform_data(data, metadata={})
        result = scaler.inverse_transform({"features": transformed})
        assert isinstance(result, ak.Array)
