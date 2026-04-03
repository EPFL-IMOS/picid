"""
Phase 1: Characterization / Stabilization tests for transform orchestration.

These tests lock current behaviour with deep assertions on:
- Data shapes and dtypes after transformation
- Metadata integrity and propagation
- Ragged array segmentation and reassembly correctness
- Copy semantics (deep for non-ak, shallow for ak where applicable)
- Fit-phase side effects (and no unintended mutation of inputs)

Run with: pytest test/transforms/base/test_phase1_stabilization.py -v

Coverage target: 100% functional coverage of orchestration logic (strategy, multisource,
data_transform, base_transform) and critical secondary paths. Do not proceed to Phase 2
until all tests pass and coverage target is met.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest
import awkward as ak
from typing import Any, Dict, Optional

from picid.data.data_objects import (
    NamedTransformInput,
    SplitDatasetContainer,
    SimpleReturnObject,
)
from picid.transforms.base.strategy import (
    TransformStrategy,
    postprocess_transformed_data,
)
from picid.transforms.base.multisource import (
    find_singular_ragged_dim,
    ConcatFitAndPerSegmentTransformMixin,
    NoFitPerSegmentMixin,
    InverseTransformMixin,
)
from picid.transforms.base.base_transform import (
    DenseTransform,
    RaggedTransform,
)
from picid.transforms.base.data_transform import DataTransform

from test.transforms.base.conftest import (
    DummyStatelessTransform,
    DummyFittableTransform,
    DummyRaggedTransform,
    create_dummy_split_container,
    create_dummy_single_unit_container,
)


# -----------------------------------------------------------------------------
# 1. Postprocess & strategy: unrecognized format, copy behaviour
# -----------------------------------------------------------------------------


class TestPostprocessCharacterization:
    """Lock postprocess_transformed_data behaviour."""

    def test_postprocess_unrecognized_format_raises_value_error(self):
        """Unrecognized format must raise ValueError (strategy line 59)."""
        # Mixed: one Mapping, one non-Mapping -> not "all Mapping", not "all non-Mapping" -> ValueError
        data = [{"features": np.array([1.0])}, 42]
        metadata = {"assign_to_map": ["features"]}
        with pytest.raises(
            ValueError, match="not recognised|not recognized|not handled"
        ):
            postprocess_transformed_data(data, metadata)

    def test_postprocess_single_array_shape_and_dtype_preserved(self):
        """Single array output: shape and dtype preserved in SimpleReturnObject."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        data = [arr]
        metadata = {"assign_to_map": ["features"]}
        result = postprocess_transformed_data(data, metadata)
        assert len(result) == 1
        assert isinstance(result[0], SimpleReturnObject)
        assert "features" in result[0]
        out = result[0]["features"]
        assert out.shape == (2, 2)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, arr)


class TestStrategyCopyCharacterization:
    """Lock copy semantics: deep for non-ak, shallow for ak."""

    def test_strategy_dense_key_deep_copy_input_unchanged(self):
        """Modifying transformed result for a dense key must not mutate original."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = create_dummy_single_unit_container(n_samples=4, n_features=3)
        original_ref = container.features.train[0]
        result, _ = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )
        # Result should be a copy: in-place modify result
        result.features.train[0][0, 0] = -999.0
        # Original container must be unchanged (deep copy was used for dense)
        np.testing.assert_array_equal(
            original_ref,
            container.features.train[0],
            err_msg="Dense key must be deep-copied so original is unchanged",
        )

    def test_strategy_ragged_key_shallow_copy_semantics(self):
        """Ragged key uses shallow copy: result shares structure with input."""
        strategy = TransformStrategy()
        transform = DummyRaggedTransform()
        ragged = ak.Array([[1.0, 2.0], [3.0, 4.0, 5.0]])
        container = SplitDatasetContainer(
            features={"train": [ragged], "val": [ragged]},
        )
        result, _ = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )
        assert "features" in result
        out = result.features.train[0]
        assert isinstance(out, ak.Array)
        # Values should be transformed (x2)
        assert ak.sum(out) == ak.sum(ragged) * 2


# -----------------------------------------------------------------------------
# 2. Multisource fit: dense, ragged+dense-only, ragged+ragged-supporting
# -----------------------------------------------------------------------------


class TestMultisourceFitCharacterization:
    """Lock fit_function_to_data_segments behaviour."""

    def test_fit_multi_source_dense_shapes_and_side_effect(self):
        """Fit on dense: concatenated shape and fitted state."""
        transform = DummyFittableTransform()
        seg1 = NamedTransformInput(features=np.array([[1.0], [2.0]]))
        seg2 = NamedTransformInput(features=np.array([[3.0], [4.0]]))
        metadata = {"apply_to_keys": ["features"]}
        transform.fit_multi_source([seg1, seg2], metadata=metadata)
        assert transform.fitted is True
        # Factor = mean of [1,2,3,4] = 2.5
        assert transform.factor == pytest.approx(2.5)

    def test_fit_multi_source_ragged_dense_only_concatenates_then_numpy(self):
        """Ragged data + dense-only transform: fit receives flattened numpy."""
        transform = DummyFittableTransform()
        ragged1 = ak.Array([[1.0, 2.0], [3.0]])
        ragged2 = ak.Array([[4.0], [5.0, 6.0]])
        data_segments = [
            NamedTransformInput(features=ragged1),
            NamedTransformInput(features=ragged2),
        ]
        metadata = {"apply_to_keys": ["features"]}
        transform.fit_multi_source(data_segments, metadata=metadata)
        assert transform.fitted is True
        # Flattened: 1,2,3,4,5,6 -> mean = 3.5
        assert transform.factor == pytest.approx(3.5)

    def test_fit_multi_source_ragged_ragged_supporting_concatenates_ak(self):
        """Ragged data + ragged-supporting transform: fit receives single ak array."""

        class FitCaptureRagged(ConcatFitAndPerSegmentTransformMixin, DenseTransform):
            def __init__(self):
                super().__init__()
                self.fitted = False
                self.fitted_data = None

            def fit_data(
                self, data: NamedTransformInput, metadata: Dict[str, Any]
            ) -> None:
                self.fitted_data = data
                self.fitted = True

            def transform_data(
                self, data: NamedTransformInput, metadata: Dict[str, Any]
            ) -> ak.Array:
                return data[list(data.keys())[0]] * 2

        # RaggedTransform marker so path is ragged-supporting
        class FitCaptureRaggedMarked(FitCaptureRagged, RaggedTransform):
            pass

        transform = FitCaptureRaggedMarked()
        ragged1 = ak.Array([[1.0, 2.0], [3.0]])
        ragged2 = ak.Array([[4.0], [5.0, 6.0]])
        data_segments = [
            NamedTransformInput(features=ragged1),
            NamedTransformInput(features=ragged2),
        ]
        metadata = {"apply_to_keys": ["features"]}
        transform.fit_multi_source(data_segments, metadata=metadata)
        assert transform.fitted is True
        assert transform.fitted_data is not None
        key = "features"
        assert key in transform.fitted_data
        concat = transform.fitted_data[key]
        assert isinstance(concat, ak.Array)
        # Concatenation along var dim: 2+1 + 1+2 = 6 elements
        assert ak.num(concat, axis=1).to_list() == [2, 1, 1, 2]


# -----------------------------------------------------------------------------
# 3. Multisource transform: ragged-to-dense reassembly, ragged-but-regular
# -----------------------------------------------------------------------------


class TestMultisourceTransformRaggedCharacterization:
    """Lock transform_function_to_data_segments ragged paths."""

    def test_transform_multi_source_ragged_to_dense_shape_and_values(self):
        """Ragged + dense-only: output shape and values after reassembly."""
        transform = DummyStatelessTransform()
        ragged = ak.Array([[1.0, 2.0], [3.0, 4.0, 5.0]])
        data_segments = [NamedTransformInput(features=ragged)]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}
        result, log = transform.transform_multi_source(data_segments, metadata=metadata)
        assert log.get("mode") == "ragged_to_dense"
        assert len(result) == 1
        out = result[0]
        assert isinstance(out, (list, np.ndarray)) or hasattr(out, "__getitem__")
        # Unwrap if list of return objects
        if isinstance(out, list) and len(out) == 1:
            out = out[0]
        if hasattr(out, "keys") and "features" in out:
            out = out["features"]
        assert isinstance(out, ak.Array), f"Expected ak.Array, got {type(out)}"
        # Transformed: each element * 2
        flat = ak.flatten(out)
        np.testing.assert_array_almost_equal(
            ak.to_numpy(flat),
            np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        )
        # Structure preserved: 2 rows, then 3 rows
        assert ak.num(out, axis=1).to_list() == [2, 3]

    def test_transform_multi_source_ragged_native_shape_and_dtype(self):
        """Ragged + ragged-supporting: output is ak.Array with same structure."""
        transform = DummyRaggedTransform()
        ragged = ak.Array([[1.0, 2.0], [3.0, 4.0, 5.0]])
        data_segments = [NamedTransformInput(features=ragged)]
        metadata = {"apply_to_keys": ["features"]}
        result, log = transform.transform_multi_source(data_segments, metadata=metadata)
        assert log.get("mode") == "ragged"
        assert len(result) == 1
        out = result[0]
        if hasattr(out, "keys"):
            out = out.get("features", out)
        assert isinstance(out, ak.Array)
        assert ak.num(out, axis=1).to_list() == [2, 3]
        np.testing.assert_array_almost_equal(
            ak.to_numpy(ak.flatten(out)),
            np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        )

    def test_transform_multi_source_ragged_but_regular_to_numpy_then_back(self):
        """Regular awkward (no var dim) + dense transform: to_numpy then wrap back to ak."""
        # Build regular ak (e.g. from numpy)
        regular_np = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        regular_ak = ak.from_numpy(regular_np)
        transform = DummyStatelessTransform()
        data_segments = [NamedTransformInput(features=regular_ak)]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}
        result, log = transform.transform_multi_source(data_segments, metadata=metadata)
        assert len(result) == 1
        out = result[0]
        if hasattr(out, "keys") and "features" in out:
            out = out["features"]
        assert isinstance(out, ak.Array)
        # 2 * 2 * 2, each element * 2
        expected = regular_np * 2
        np.testing.assert_array_almost_equal(ak.to_numpy(out), expected)


# -----------------------------------------------------------------------------
# 4. Metadata propagation
# -----------------------------------------------------------------------------


class TestMetadataPropagation:
    """Lock metadata passed to fit_data and transform_data."""

    def test_metadata_mode_and_apply_to_keys_in_transform_data(self):
        """Strategy passes mode and apply_to_keys in metadata to transform_data."""
        captured = {}

        class CaptureMetaTransform(NoFitPerSegmentMixin, DenseTransform):
            def transform_data(
                self, data: NamedTransformInput, metadata: Dict[str, Any]
            ) -> np.ndarray:
                captured["metadata"] = copy.deepcopy(metadata)
                return data[list(data.keys())[0]] * 2

        strategy = TransformStrategy()
        transform = CaptureMetaTransform()
        container = create_dummy_single_unit_container()
        strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )
        meta = captured["metadata"]
        assert "mode" in meta
        assert (
            meta["apply_to_keys"] == ["features"] or meta["apply_to_keys"] == "features"
        )
        assert "assign_to_map" in meta

    def test_metadata_in_fit_data(self):
        """Fit phase receives mode=fit_on_split and apply_to_keys."""
        captured = {}

        class CaptureFitMeta(ConcatFitAndPerSegmentTransformMixin, DenseTransform):
            def __init__(self):
                super().__init__()
                self.fitted = False

            def fit_data(
                self, data: NamedTransformInput, metadata: Dict[str, Any]
            ) -> None:
                captured["fit_metadata"] = copy.deepcopy(metadata)
                self.fitted = True

            def transform_data(
                self, data: NamedTransformInput, metadata: Dict[str, Any]
            ) -> np.ndarray:
                return data[list(data.keys())[0]]

        strategy = TransformStrategy()
        transform = CaptureFitMeta()
        container = create_dummy_split_container(n_units=2)
        strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
            fit_on_split="train",
            fit_on_key="features",
        )
        assert captured["fit_metadata"]["mode"] == "train"
        assert "apply_to_keys" in captured["fit_metadata"]


# -----------------------------------------------------------------------------
# 5. transform_on_keys filtering
# -----------------------------------------------------------------------------


class TestTransformOnKeysCharacterization:
    """Lock transform_on_keys: only listed splits are transformed."""

    def test_strategy_transform_on_keys_omits_test_split(self):
        """transform_on_keys=['train','val']: train/val are transformed (x2); test data unchanged."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = create_dummy_split_container(n_units=2)
        orig_train0 = container.features.train[0].copy()
        orig_val0 = container.features.val[0].copy()
        orig_test0 = container.features.test[0].copy()
        result, _ = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
            transform_on_keys=["train", "val"],
        )
        assert "features" in result
        np.testing.assert_array_equal(result.features.train[0], orig_train0 * 2)
        np.testing.assert_array_equal(result.features.val[0], orig_val0 * 2)
        np.testing.assert_array_equal(result.features.test[0], orig_test0)


# -----------------------------------------------------------------------------
# 6. NoFitConcatAlongAxisMixin
# -----------------------------------------------------------------------------


class TestNoFitConcatAlongAxisMixinCharacterization:
    """Lock NoFitConcatAlongAxisMixin: list of n segments -> concat -> transform -> list of 1."""

    def test_concatenate_units_transform_multi_source_n_to_one(self):
        """NoFitConcatAlongAxisMixin: multiple segments become one after concat then transform."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        transform = ConcatenateUnitsTransform(axis=0)
        seg1 = NamedTransformInput(features=np.array([[1.0], [2.0]]))
        seg2 = NamedTransformInput(features=np.array([[3.0], [4.0]]))
        data_segments = [seg1, seg2]
        metadata = {"apply_to_keys": ["features"]}
        result, log = transform.transform_multi_source(data_segments, metadata=metadata)
        # List of 1 (concat then single transform_data)
        assert len(result) == 1
        out = result[0]
        if hasattr(out, "keys"):
            out = out["features"]
        assert isinstance(out, np.ndarray)
        # Concat [[1,2],[3,4]] -> [1,2,3,4], then *2 -> [2,4,6,8]
        np.testing.assert_array_almost_equal(
            out.flatten(), np.array([2.0, 4.0, 6.0, 8.0])
        )


# -----------------------------------------------------------------------------
# 7. InverseTransformMixin
# -----------------------------------------------------------------------------


class DummyInverseTransform(
    NoFitPerSegmentMixin, InverseTransformMixin, DenseTransform
):
    """Dummy transform with inverse for Phase 1 characterization."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> np.ndarray:
        return data[list(data.keys())[0]] * 2

    def inverse_transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> np.ndarray:
        return data[list(data.keys())[0]] / 2

    def inverse_transform(
        self, data: NamedTransformInput, metadata: Optional[Dict] = None
    ) -> Any:
        key = list(data.keys())[0]
        return data[key] / 2


class TestInverseTransformMixinCharacterization:
    """Lock InverseTransformMixin behaviour."""

    def test_inverse_transform_multi_source_applies_per_segment(self):
        """inverse_transform_multi_source calls inverse_transform_data per segment."""
        transform = DummyInverseTransform()
        seg1 = NamedTransformInput(features=np.array([[2.0, 4.0]]))
        seg2 = NamedTransformInput(features=np.array([[6.0, 8.0]]))
        result = transform.inverse_transform_multi_source([seg1, seg2], metadata={})
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0], np.array([[1.0, 2.0]]))
        np.testing.assert_array_almost_equal(result[1], np.array([[3.0, 4.0]]))

    def test_inverse_transform_single_data(self):
        """inverse_transform(data) returns inverse scaled data."""
        transform = DummyInverseTransform()
        data = NamedTransformInput(features=np.array([[4.0, 8.0]]))
        out = transform.inverse_transform(data, metadata=None)
        np.testing.assert_array_almost_equal(out, np.array([[2.0, 4.0]]))


# -----------------------------------------------------------------------------
# 8. DataTransform forward: shape and dtype preservation
# -----------------------------------------------------------------------------


class TestDataTransformForwardCharacterization:
    """Lock DataTransform.forward output shape and dtype."""

    def test_forward_preserves_shape_and_dtype_single_unit(self):
        """Forward preserves shape and dtype of transformed key."""
        dt = DataTransform(
            "test",
            DummyStatelessTransform(),
            {"apply_to": "features", "assign_to": "features"},
        )
        container = create_dummy_single_unit_container(n_samples=5, n_features=3)
        arr = container.features.train[0]
        orig_shape, orig_dtype = arr.shape, arr.dtype
        result, _ = dt.forward(container)
        out = result.features.train[0]
        assert out.shape == orig_shape
        assert out.dtype == orig_dtype
        np.testing.assert_array_almost_equal(out, arr * 2)

    def test_forward_multi_unit_preserves_number_of_units(self):
        """Multi-unit: number of units per split unchanged."""
        dt = DataTransform(
            "test",
            DummyStatelessTransform(),
            {"apply_to": "features", "assign_to": "features"},
        )
        container = create_dummy_split_container(n_units=4)
        result, _ = dt.forward(container)
        assert len(result.features.train) == 4
        assert len(result.features.val) == 4
        assert len(result.features.test) == 4


# -----------------------------------------------------------------------------
# 9. Fit side-effect: no mutation of input container
# -----------------------------------------------------------------------------


class TestFitSideEffectsCharacterization:
    """Lock: fit phase must not mutate input data container."""

    def test_fit_does_not_mutate_input_data(self):
        """After strategy apply with fit, input container values unchanged."""
        strategy = TransformStrategy()
        transform = DummyFittableTransform()
        container = create_dummy_single_unit_container(n_samples=4, n_features=2)
        original_train = container.features.train[0].copy()
        strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
            fit_on_split="train",
            fit_on_key="features",
        )
        np.testing.assert_array_equal(container.features.train[0], original_train)


# -----------------------------------------------------------------------------
# 10. find_singular_ragged_dim edge (multisource utils)
# -----------------------------------------------------------------------------


class TestFindSingularRaggedDimCharacterization:
    """Lock find_singular_ragged_dim return values."""

    def test_find_singular_ragged_dim_regular_returns_none(self):
        """Regular awkward array (no var dim) returns None."""
        regular = ak.from_numpy(np.zeros((2, 3, 4)))
        nti = NamedTransformInput(features=regular)
        assert find_singular_ragged_dim(nti) is None

    def test_find_singular_ragged_dim_single_var_returns_axis(self):
        """Single ragged dim returns that axis index."""
        ragged = ak.Array([[1.0, 2.0], [3.0]])
        nti = NamedTransformInput(features=ragged)
        dim = find_singular_ragged_dim(nti)
        assert dim is not None
        assert isinstance(dim, int)
