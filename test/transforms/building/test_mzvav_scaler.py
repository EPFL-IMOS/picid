"""Tests for picid.transforms.building.mzvav_scaler."""

from __future__ import annotations

import numpy as np
import awkward as ak
import pytest

from picid.data.data_objects import NamedTransformInput
from picid.transforms.building.mzvav_scaler import MinMaxScalerMZVAV


def _make_data(values: np.ndarray) -> dict:
    """Wrap a 3-D numpy array as the single-key NamedTransformInput expected by the scaler."""
    return {"features": ak.from_numpy(values)}


def _make_ragged(
    timesteps: list[int], n_features: int, rng: np.random.Generator
) -> ak.Array:
    """Build a ``n_days × var_timesteps × n_features`` ragged Awkward array.

    Only the middle (time) dimension varies; the inner feature dimension is
    fixed — matching the actual MZVAV pipeline shape.
    """
    parts = [rng.random((t, n_features)).astype(np.float32) for t in timesteps]
    all_data = np.concatenate(parts, axis=0)
    offsets = np.concatenate([[0], np.cumsum(np.array(timesteps))])
    content = ak.from_numpy(all_data)
    layout = ak.contents.ListOffsetArray(ak.index.Index64(offsets), content.layout)
    return ak.Array(layout)


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


@pytest.mark.unit
class TestMinMaxScalerMZVAVRaggedPath:
    """Tests for the ragged multi-source dispatch path.

    The actual MZVAV pipeline emits data of shape ``n_days × var_timesteps × 17``
    (only the time dimension varies across days).  ``fit_multi_source`` and
    ``transform_multi_source`` route through ``RaggedDenseHandler``, which
    flattens each day into a 2-D slice before calling ``fit_data`` /
    ``transform_data``.  These tests exercise that dispatch to guard against
    regressions in the ragged code path.
    """

    _N_FEATURES = 17
    _TRAIN_DAYS = [10, 15, 8]
    _VAL_DAYS = [9, 11]
    _METADATA = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}

    def _make_segments(self, rng: np.random.Generator):
        train_seg = NamedTransformInput(
            features=_make_ragged(self._TRAIN_DAYS, self._N_FEATURES, rng)
        )
        val_seg = NamedTransformInput(
            features=_make_ragged(self._VAL_DAYS, self._N_FEATURES, rng)
        )
        return train_seg, val_seg

    def test_ragged_array_has_expected_type(self):
        """Helper produces the same ragged shape as the MZVAV pipeline."""
        rng = np.random.default_rng(0)
        arr = _make_ragged(self._TRAIN_DAYS, self._N_FEATURES, rng)
        assert (
            str(arr.type)
            == f"{len(self._TRAIN_DAYS)} * var * {self._N_FEATURES} * float32"
        )

    def test_fit_multi_source_does_not_raise(self):
        rng = np.random.default_rng(1)
        train_seg, _ = self._make_segments(rng)
        scaler = MinMaxScalerMZVAV()
        scaler.fit_multi_source([train_seg], metadata=self._METADATA)

    def test_transform_multi_source_output_length_matches_input(self):
        rng = np.random.default_rng(2)
        train_seg, val_seg = self._make_segments(rng)
        scaler = MinMaxScalerMZVAV()
        scaler.fit_multi_source([train_seg], metadata=self._METADATA)
        out, _ = scaler.transform_multi_source(
            [train_seg, val_seg], metadata=self._METADATA
        )
        assert len(out) == 2

    def test_transform_multi_source_train_scaled_to_unit_range(self):
        """Train segment values must lie in [0, 1] after scaling."""
        rng = np.random.default_rng(3)
        train_seg, _ = self._make_segments(rng)
        scaler = MinMaxScalerMZVAV()
        scaler.fit_multi_source([train_seg], metadata=self._METADATA)
        out, _ = scaler.transform_multi_source([train_seg], metadata=self._METADATA)
        flat = ak.to_numpy(ak.flatten(out[0]["features"], axis=None))
        assert flat.min() >= -1e-6, f"Expected min >= 0, got {flat.min()}"
        assert flat.max() <= 1.0 + 1e-6, f"Expected max <= 1, got {flat.max()}"

    def test_transform_multi_source_preserves_day_count(self):
        """Output must have the same number of 'days' (outer dim) as input."""
        rng = np.random.default_rng(4)
        train_seg, _ = self._make_segments(rng)
        scaler = MinMaxScalerMZVAV()
        scaler.fit_multi_source([train_seg], metadata=self._METADATA)
        out, _ = scaler.transform_multi_source([train_seg], metadata=self._METADATA)
        assert len(out[0]["features"]) == len(self._TRAIN_DAYS)

    def test_transform_multi_source_preserves_per_day_timesteps(self):
        """Each day's timestep count must be unchanged after transform."""
        rng = np.random.default_rng(5)
        train_seg, _ = self._make_segments(rng)
        scaler = MinMaxScalerMZVAV()
        scaler.fit_multi_source([train_seg], metadata=self._METADATA)
        out, _ = scaler.transform_multi_source([train_seg], metadata=self._METADATA)
        out_feat = out[0]["features"]
        for i, expected_t in enumerate(self._TRAIN_DAYS):
            assert (
                len(out_feat[i]) == expected_t
            ), f"Day {i}: expected {expected_t} timesteps, got {len(out_feat[i])}"

    @pytest.mark.xfail(
        reason=(
            "inverse_transform_multi_source passes the full ragged segment directly "
            "to inverse_transform, which cannot convert a ragged ak.Array to numpy. "
            "This path is never exercised in the MZVAV pipeline because all evaluator "
            "configs set apply_inverse_scaling=false."
        ),
        strict=True,
    )
    def test_inverse_transform_multi_source_with_ragged_is_known_limitation(self):
        """Document that calling inverse_transform_multi_source on ragged data fails."""
        rng = np.random.default_rng(6)
        train_seg, _ = self._make_segments(rng)
        scaler = MinMaxScalerMZVAV()
        scaler.fit_multi_source([train_seg], metadata=self._METADATA)
        out, _ = scaler.transform_multi_source([train_seg], metadata=self._METADATA)
        # This raises TypeError because inverse_transform cannot handle ragged arrays.
        scaler.inverse_transform_multi_source(out, metadata=self._METADATA)
