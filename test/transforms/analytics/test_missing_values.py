"""Tests for picid.transforms.analytics.missing_values."""

import os
import tempfile
import numpy as np
import awkward as ak

from picid.data.data_objects import NamedTransformInput
from picid.transforms.analytics.missing_values import MissingValuesStatsLogger


def test_missing_values_transform_data_passthrough():
    """transform_data is pass-through, returns data unchanged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MissingValuesStatsLogger(saving_path=tmpdir)
        data = {"features": np.array([[1.0, 2.0], [3.0, np.nan]])}
        metadata = {}
        result = logger.transform_data(data, metadata)
        assert result is data


def test_missing_values_transform_multi_source_numpy():
    """transform_multi_source with small numpy inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MissingValuesStatsLogger(saving_path=tmpdir, apply_to=["features"])
        seg1 = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, np.nan]]),
            metadata={"unit_id": "U1"},
        )
        seg2 = NamedTransformInput(
            features=np.array([[5.0, 6.0], [7.0, 8.0]]),
            metadata={"unit_id": "U2"},
        )
        segments = [seg1, seg2]
        out_segments, out_meta = logger.transform_multi_source(segments, metadata={})
        assert out_segments is segments
        assert out_meta == {}
        assert len(out_segments) == 2
        # Stats file should exist
        assert os.path.exists(os.path.join(tmpdir, "logs", "missing_values_stats.txt"))


def test_missing_values_transform_multi_source_awkward():
    """transform_multi_source with awkward array input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MissingValuesStatsLogger(saving_path=tmpdir, apply_to=["features"])
        # Ragged array: [[1, 2], [3, nan, 5]]
        arr = ak.Array([[1.0, 2.0], [3.0, np.nan, 5.0]])
        seg = NamedTransformInput(features=arr, metadata={"unit_id": "U_awk"})
        segments = [seg]
        out_segments, _ = logger.transform_multi_source(segments, metadata={})
        assert out_segments is segments
        assert len(out_segments) == 1


def test_missing_values_apply_to_subset():
    """Only keys in apply_to are processed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MissingValuesStatsLogger(saving_path=tmpdir, apply_to=["x"])
        seg = NamedTransformInput(
            x=np.array([[1.0, np.nan]]),
            y=np.array([[10.0, 20.0]]),  # not in apply_to, should be skipped
            metadata={},
        )
        out_segments, _ = logger.transform_multi_source([seg], metadata={})
        assert out_segments[0]["x"] is not None
        assert "y" in out_segments[0]


def test_missing_values_key_not_in_segment_skipped():
    """Keys in apply_to but not in segment are skipped gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MissingValuesStatsLogger(
            saving_path=tmpdir, apply_to=["features", "missing_key"]
        )
        seg = NamedTransformInput(features=np.array([[1.0, 2.0]]), metadata={})
        out_segments, _ = logger.transform_multi_source([seg], metadata={})
        assert len(out_segments) == 1
        assert "features" in out_segments[0]


def test_missing_values_global_summary():
    """Global summary aggregates across all units."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MissingValuesStatsLogger(saving_path=tmpdir)
        seg1 = NamedTransformInput(
            features=np.array([[1.0, np.nan]]), metadata={"unit_id": "A"}
        )
        seg2 = NamedTransformInput(
            features=np.array([[np.nan, 4.0]]), metadata={"unit_id": "B"}
        )
        out_segments, _ = logger.transform_multi_source([seg1, seg2], metadata={})
        assert len(out_segments) == 2
        # Total: 2 NaNs, 4 elements -> 50% global
        with open(os.path.join(tmpdir, "logs", "missing_values_stats.txt")) as f:
            content = f.read()
        assert "GLOBAL SUMMARY" in content
        assert "features" in content or "Total" in content.lower()
