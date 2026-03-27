"""Tests for Phase 2.1: DataKind, TransformCapability, infer_data_kind, get_capability."""

import numpy as np
import awkward as ak

from picid.transforms.base.data_kind import (
    DATA_KIND_DENSE,
    DATA_KIND_RAGGED,
    DATA_KIND_RAGGED_REGULAR,
    CAPABILITY_DENSE,
    CAPABILITY_RAGGED,
    CAPABILITY_BOTH,
    get_capability,
    infer_data_kind,
)
from picid.transforms.base.multisource import find_singular_ragged_dim
from picid.data.data_objects import NamedTransformInput

from test.transforms.base.conftest import (
    DummyStatelessTransform,
    DummyRaggedTransform,
    DummyRaggedOrDenseTransform,
)


class TestGetCapability:
    def test_dense_transform_returns_dense(self):
        assert get_capability(DummyStatelessTransform()) == CAPABILITY_DENSE

    def test_ragged_transform_returns_ragged(self):
        assert get_capability(DummyRaggedTransform()) == CAPABILITY_RAGGED

    def test_ragged_or_dense_returns_both(self):
        assert get_capability(DummyRaggedOrDenseTransform()) == CAPABILITY_BOTH


class TestInferDataKind:
    def test_dense_segments_returns_dense(self):
        segments = [NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))]
        kind = infer_data_kind(segments, ["features"], find_singular_ragged_dim)
        assert kind == DATA_KIND_DENSE

    def test_ragged_segments_returns_ragged(self):
        segments = [NamedTransformInput(features=ak.Array([[1.0, 2.0], [3.0]]))]
        kind = infer_data_kind(segments, ["features"], find_singular_ragged_dim)
        assert kind == DATA_KIND_RAGGED

    def test_ragged_regular_segments_returns_ragged_regular(self):
        # Regular awkward (no variable-length dim)
        regular_ak = ak.from_numpy(np.ones((2, 3, 4)))
        segments = [NamedTransformInput(features=regular_ak)]
        kind = infer_data_kind(segments, ["features"], find_singular_ragged_dim)
        assert kind == DATA_KIND_RAGGED_REGULAR

    def test_apply_to_keys_string_normalized(self):
        segments = [NamedTransformInput(features=np.array([[1.0]]))]
        kind = infer_data_kind(segments, "features", find_singular_ragged_dim)
        assert kind == DATA_KIND_DENSE

    def test_empty_segments_returns_dense(self):
        kind = infer_data_kind([], ["features"], find_singular_ragged_dim)
        assert kind == DATA_KIND_DENSE
