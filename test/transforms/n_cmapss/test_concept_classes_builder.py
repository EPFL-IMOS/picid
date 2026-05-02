"""Tests for picid.transforms.n_cmapss.concept_classes_builder module.

Coverage target: >=95% of picid/transforms/n_cmapss/concept_classes_builder.py

Tests cover the ClassLabelLookup registry and ConceptClassesBuilder
fit/transform pipeline, including binary concept validation, class label
assignment, and unit-ID-based class multiplication.
"""

import logging

import numpy as np
import pytest

from picid.data.data_objects import NamedTransformInput
from picid.transforms.n_cmapss.concept_classes_builder import (
    ClassLabelLookup,
    ConceptClassesBuilder,
)


@pytest.mark.unit
class TestClassLabelLookup:
    """Tests for ClassLabelLookup — sequential integer label registry."""

    def test_get_missing_key_returns_none(self):
        """Unknown key returns None.

        **Methodology**: Query an empty lookup.

        **Expected**: None returned.
        """
        lookup = ClassLabelLookup()
        assert lookup.get("unknown") is None

    def test_set_assigns_sequential_labels(self):
        """Keys are assigned labels 1, 2, 3 in order.

        **Methodology**: Register three distinct keys.

        **Expected**: Labels are 1, 2, 3 respectively.
        """
        lookup = ClassLabelLookup()
        assert lookup.set("a") == 1
        assert lookup.set("b") == 2
        assert lookup.set("c") == 3

    def test_set_idempotent_for_existing_key(self):
        """Re-registering a key returns the same label.

        **Methodology**: Set "a" twice.

        **Expected**: Same label both times; counter not incremented.
        """
        lookup = ClassLabelLookup()
        first = lookup.set("a")
        second = lookup.set("a")
        assert first == second == 1
        assert lookup.counter == 2

    def test_counter_starts_at_one(self):
        """First assigned label is 1, not 0.

        **Methodology**: Fresh lookup, set one key.

        **Expected**: Label is 1.
        """
        lookup = ClassLabelLookup()
        assert lookup.set("first") == 1

    def test_get_after_set_returns_label(self):
        """get() returns the label assigned by set().

        **Methodology**: Set then get the same key.

        **Expected**: Same integer value.
        """
        lookup = ClassLabelLookup()
        label = lookup.set("x")
        assert lookup.get("x") == label


@pytest.mark.unit
class TestConceptClassesBuilder:
    """Tests for ConceptClassesBuilder fit/transform pipeline."""

    def _make_builder_fitted(self, ds_values):
        """Create and fit a builder with the given DS values."""
        builder = ConceptClassesBuilder()
        fit_data = NamedTransformInput(
            n_DS=np.array(ds_values, dtype=float),
            concepts=np.zeros((len(ds_values), 1)),
        )
        builder.fit_data(fit_data, {})
        return builder

    def test_fit_registers_unique_ds_values(self):
        """fit_data registers each unique n_DS value in the lookup.

        **Methodology**: Fit on n_DS=[1,1,2,2,3].

        **Expected**: Lookup has entries for "n_DS_1", "n_DS_2", "n_DS_3".
        """
        builder = self._make_builder_fitted([1, 1, 2, 2, 3])
        assert builder.lookup.get("n_DS_1") is not None
        assert builder.lookup.get("n_DS_2") is not None
        assert builder.lookup.get("n_DS_3") is not None

    def test_fit_idempotent_for_repeated_ds(self):
        """Repeated DS values produce a single lookup entry.

        **Methodology**: Fit on all-1s n_DS.

        **Expected**: One entry in lookup, counter=2.
        """
        builder = self._make_builder_fitted([1, 1, 1, 1])
        assert builder.lookup.get("n_DS_1") == 1
        assert builder.lookup.counter == 2

    def test_single_active_concept(self):
        """Single active concept column maps to correct class.

        **Methodology**: 3 concept columns, column 1 active for all rows, DS=1.

        **Expected**: class = (argmax=1) + 1 = 2, multiplied by class_id=1 → 2.
        """
        builder = self._make_builder_fitted([1])
        n = 5
        concepts = np.zeros((n, 3))
        concepts[:, 1] = 1.0
        data = NamedTransformInput(
            n_DS=np.ones(n),
            concepts=concepts,
        )
        result = builder.transform_data(data, {})
        expected = np.full((n, 1), 2)
        np.testing.assert_array_equal(result["concepts"], expected)

    def test_all_zero_concepts_map_to_zero(self):
        """All-zero concepts (healthy state) map to class 0.

        **Methodology**: No active concepts, DS=1.

        **Expected**: All classes are 0.
        """
        builder = self._make_builder_fitted([1])
        n = 5
        data = NamedTransformInput(
            n_DS=np.ones(n),
            concepts=np.zeros((n, 3)),
        )
        result = builder.transform_data(data, {})
        np.testing.assert_array_equal(result["concepts"], np.zeros((n, 1)))

    def test_mixed_concepts_correct_mapping(self):
        """Mixed rows (some active, some zero) are classified correctly.

        **Methodology**: Row 0 has concept 0 active, row 1 all zero, row 2 has concept 2.

        **Expected**: Classes [1, 0, 3] (after +1, before class_id multiplication).
        """
        builder = self._make_builder_fitted([1])
        concepts = np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        data = NamedTransformInput(
            n_DS=np.ones(3),
            concepts=concepts,
        )
        result = builder.transform_data(data, {})
        np.testing.assert_array_equal(result["concepts"], np.array([[1], [0], [3]]))

    def test_class_multiplied_by_ds_lookup_id(self):
        """Classes are multiplied by the DS lookup ID.

        **Methodology**: Fit on DS=[5, 10] (lookup assigns 1 and 2).
        Transform with DS=10 (class_id=2).

        **Expected**: Base class 1 → 1*2 = 2.
        """
        builder = self._make_builder_fitted([5, 10])
        class_id_for_10 = builder.lookup.get("n_DS_10")
        assert class_id_for_10 == 2

        concepts = np.array([[1, 0, 0]], dtype=float)
        data = NamedTransformInput(
            n_DS=np.array([10.0]),
            concepts=concepts,
        )
        result = builder.transform_data(data, {})
        np.testing.assert_array_equal(
            result["concepts"], np.array([[1 * class_id_for_10]])
        )

    def test_non_binary_concepts_rounded_with_warning(self, caplog):
        """Non-binary concept values are rounded and a warning is logged.

        **Methodology**: Supply concepts with value 0.7 (rounds to 1).

        **Expected**: Values rounded, warning about non-binary logged.
        """
        builder = self._make_builder_fitted([1])
        concepts = np.array([[0.7, 0.0, 0.0]], dtype=float)
        data = NamedTransformInput(
            n_DS=np.ones(1),
            concepts=concepts,
        )
        with caplog.at_level(logging.WARNING):
            result = builder.transform_data(data, {})
        assert any("Non-binary" in msg for msg in caplog.messages)
        np.testing.assert_array_equal(result["concepts"], np.array([[1]]))

    def test_non_roundable_concepts_raises(self):
        """Concepts that round outside {0,1} raise AssertionError.

        **Methodology**: Supply concept value 1.6 (rint→2.0, not in {0,1}).

        **Expected**: AssertionError about non-binary.
        """
        builder = self._make_builder_fitted([1])
        concepts = np.array([[1.6, 0.0, 0.0]], dtype=float)
        data = NamedTransformInput(
            n_DS=np.ones(1),
            concepts=concepts,
        )
        with pytest.raises(AssertionError, match="Non-binary"):
            builder.transform_data(data, {})

    def test_combined_error_modes_raises(self):
        """Two active concepts in one row raises AssertionError.

        **Methodology**: Row with concepts [1, 1, 0] (sum=2 > 1).

        **Expected**: AssertionError about combined error modes.
        """
        builder = self._make_builder_fitted([1])
        concepts = np.array([[1, 1, 0]], dtype=float)
        data = NamedTransformInput(
            n_DS=np.ones(1),
            concepts=concepts,
        )
        with pytest.raises(AssertionError, match="Combined error modes"):
            builder.transform_data(data, {})

    def test_unfitted_ds_raises(self):
        """Transform with unseen DS value raises AssertionError.

        **Methodology**: Fit on DS=1, transform with DS=99.

        **Expected**: AssertionError about class index not found.
        """
        builder = self._make_builder_fitted([1])
        concepts = np.array([[1, 0, 0]], dtype=float)
        data = NamedTransformInput(
            n_DS=np.array([99.0]),
            concepts=concepts,
        )
        with pytest.raises(AssertionError, match="Class index was not found"):
            builder.transform_data(data, {})

    def test_output_shape_is_n_by_1(self):
        """Output concepts shape is (n, 1) regardless of input columns.

        **Methodology**: 5 concept columns, 4 rows.

        **Expected**: Output shape is (4, 1).
        """
        builder = self._make_builder_fitted([1])
        n = 4
        concepts = np.zeros((n, 5))
        data = NamedTransformInput(
            n_DS=np.ones(n),
            concepts=concepts,
        )
        result = builder.transform_data(data, {})
        assert result["concepts"].shape == (n, 1)

    def test_call_delegates_to_transform_data(self):
        """__call__ produces same result as transform_data.

        **Methodology**: Call builder(data) and builder.transform_data(data, None).

        **Expected**: Identical output.
        """
        builder = self._make_builder_fitted([1])
        concepts = np.array([[0, 1, 0]], dtype=float)
        data1 = NamedTransformInput(n_DS=np.ones(1), concepts=concepts.copy())
        data2 = NamedTransformInput(n_DS=np.ones(1), concepts=concepts.copy())

        result1 = builder.transform_data(data1, None)
        result2 = builder(data2)

        np.testing.assert_array_equal(result1["concepts"], result2["concepts"])
