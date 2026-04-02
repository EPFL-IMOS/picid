"""Tests for Phase 2.2: Transform handler protocol and registry."""

import numpy as np
import pytest
import awkward as ak

from picid.transforms.base.data_kind import (
    DATA_KIND_DENSE,
    DATA_KIND_RAGGED,
    DATA_KIND_RAGGED_REGULAR,
    CAPABILITY_BOTH,
    CAPABILITY_DENSE,
    CAPABILITY_RAGGED,
)
from picid.transforms.base.handlers import (
    get_handler,
    DenseDenseHandler,
    RaggedDenseHandler,
    RaggedRaggedHandler,
)
from picid.data.data_objects import NamedTransformInput

from test.transforms.base.conftest import (
    DummyStatelessTransform,
    DummyFittableTransform,
    DummyRaggedTransform,
)


class TestHandlerRegistry:
    def test_get_handler_dense_dense(self):
        h = get_handler(DATA_KIND_DENSE, CAPABILITY_DENSE)
        assert isinstance(h, DenseDenseHandler)

    def test_get_handler_ragged_dense(self):
        h = get_handler(DATA_KIND_RAGGED, CAPABILITY_DENSE)
        assert isinstance(h, RaggedDenseHandler)

    def test_get_handler_ragged_ragged(self):
        h = get_handler(DATA_KIND_RAGGED, CAPABILITY_RAGGED)
        assert isinstance(h, RaggedRaggedHandler)

    def test_get_handler_ragged_regular_dense(self):
        h = get_handler(DATA_KIND_RAGGED_REGULAR, CAPABILITY_DENSE)
        assert isinstance(h, DenseDenseHandler)

    def test_get_handler_unknown_raises(self):
        with pytest.raises(KeyError, match="No handler registered"):
            get_handler("unknown_kind", CAPABILITY_DENSE)

    def test_get_handler_dense_capability_both_uses_dense_dense(self):
        """Registry: CAPABILITY_BOTH on dense data must stay mapped to DenseDenseHandler."""
        h = get_handler(DATA_KIND_DENSE, CAPABILITY_BOTH)
        assert isinstance(h, DenseDenseHandler)

    def test_get_handler_ragged_regular_capability_both_uses_dense_dense(self):
        """Registry: regular ragged + BOTH capability uses dense path (structural compat)."""
        h = get_handler(DATA_KIND_RAGGED_REGULAR, CAPABILITY_BOTH)
        assert isinstance(h, DenseDenseHandler)

    def test_get_handler_ragged_capability_both_uses_ragged_ragged(self):
        """Registry: irregular ragged + BOTH capability uses RaggedRaggedHandler."""
        h = get_handler(DATA_KIND_RAGGED, CAPABILITY_BOTH)
        assert isinstance(h, RaggedRaggedHandler)


class TestDenseDenseHandler:
    def test_fit_prepare_concatenates_and_calls_fit(self):
        t = DummyFittableTransform()
        segments = [
            NamedTransformInput(features=np.array([[1.0], [2.0]])),
            NamedTransformInput(features=np.array([[3.0], [4.0]])),
        ]
        handler = DenseDenseHandler()
        handler.fit_prepare(
            segments,
            ["features"],
            {"apply_to_keys": ["features"]},
            lambda train, metadata: t.fit_data(NamedTransformInput(**train), metadata),
        )
        assert t.fitted
        assert t.factor == 2.5  # mean of 1,2,3,4

    def test_transform_apply_dense(self):
        t = DummyStatelessTransform()
        segments = [NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))]
        handler = DenseDenseHandler()
        out, log = handler.transform_apply(
            segments,
            t.transform_data,
            {"apply_to_keys": ["features"], "assign_to_map": ["features"]},
            t.__class__.__name__,
        )
        assert log["mode"] == "dense"
        assert len(out) == 1
        # transform_data returns ndarray (no assign_to_map postprocess in handler)
        np.testing.assert_allclose(out[0], np.array([[2.0, 4.0], [6.0, 8.0]]))


class TestRaggedRaggedHandler:
    def test_transform_apply_ragged_native(self):
        t = DummyRaggedTransform()
        segments = [NamedTransformInput(features=ak.Array([[1.0, 2.0], [3.0]]))]
        handler = RaggedRaggedHandler()
        out, log = handler.transform_apply(
            segments,
            t.transform_data,
            {"apply_to_keys": ["features"]},
            t.__class__.__name__,
        )
        assert log["mode"] == "ragged"
        assert len(out) == 1
        assert isinstance(out[0], ak.Array)
        assert ak.almost_equal(out[0], ak.Array([[2.0, 4.0], [6.0]]))
