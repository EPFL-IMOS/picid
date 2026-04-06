"""High-signal bug sentinels for foundational contracts (cross-cutting).

These tests encode invariants that are easy to regress silently: split counts,
target semantics, and inclusion rules for edge-case fold ids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.threew import ThreeWLoader

from test.fixtures.builders import (
    make_forecasting_target_continuing_features,
    make_synthetic_features_float32,
)
from test.fixtures.datasource_layouts import (
    make_threew_frame,
    touch_threew_instance,
    write_threew_folds,
)
from test.fixtures.rng import numpy_rs


def _layout_predefined_splits_and_labels(root: Path) -> dict[str, object]:
    """Local 3W layout helper so meta tests do not depend on another test module."""
    touch_threew_instance(root, 0, "WELL-NORMAL-0001")
    touch_threew_instance(root, 3, "WELL-FAULT-0002")
    touch_threew_instance(root, 7, "WELL-FAULT-0003")
    touch_threew_instance(root, 5, "SIMULATED_0001")
    write_threew_folds(
        root,
        [
            ("0/WELL-NORMAL-0001.csv", 0, False),
            ("3/WELL-FAULT-0002.csv", 1, False),
            ("7/WELL-FAULT-0003.csv", 3, False),
            ("5/SIMULATED_0001.csv", -1, False),
        ],
    )
    return {
        "WELL-NORMAL-0001": make_threew_frame(0),
        "WELL-FAULT-0002": make_threew_frame(3),
        "WELL-FAULT-0003": make_threew_frame(7),
        "SIMULATED_0001": make_threew_frame(5),
    }


@pytest.fixture
def threew_bug_sentinel_kwargs(tmp_path: Path) -> dict:
    """Minimal loader kwargs aligned with ``_layout_predefined_splits_and_labels``."""
    return {
        "data_dir": str(tmp_path),
        "data_name": "threew",
        "task_mode": "anomaly_detection",
        "download": False,
        "validation_fold": 0,
        "test_fold": 1,
        "include_ova": False,
        "export_event_class": True,
    }


def test_threew_predefined_folds_two_one_one_split_and_binary_val_test_targets(
    threew_bug_sentinel_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predefined 3W fold layout yields 2/1/1 train/val/test and correct binary val/test targets."""
    root = Path(threew_bug_sentinel_kwargs["data_dir"])
    instance_by_name = _layout_predefined_splits_and_labels(root)

    def _fake_reader(self, path: Path):
        return instance_by_name[path.stem].copy()

    monkeypatch.setattr(ThreeWLoader, "_read_instance_frame", _fake_reader)

    loader = ThreeWLoader(**threew_bug_sentinel_kwargs)
    loader.load_data()
    data = loader.get_data()

    assert isinstance(data, SplitDatasetContainer)
    assert len(data["features"]["train"]) == 2
    assert len(data["features"]["val"]) == 1
    assert len(data["features"]["test"]) == 1

    val_target = np.asarray(data["target"]["val"][0]).reshape(-1)
    test_target = np.asarray(data["target"]["test"][0]).reshape(-1)
    assert np.all(val_target == 0.0)
    assert np.all(test_target == 1.0)

    meta = loader.get_meta_data()
    assert meta["class_labels"]["val"] == [0]
    assert meta["class_labels"]["test"] == [3]


def test_forecasting_target_builder_keeps_rng_stream_aligned_with_feature_builder() -> (
    None
):
    """Forecasting targets must keep the post-feature RNG stream stable for a fixed seed."""
    features = make_synthetic_features_float32(seed=7, time_steps=4, n_features=3)
    target = make_forecasting_target_continuing_features(features, seed=7)
    rs = numpy_rs(7)
    rs.rand(*features.shape)
    expected_target = rs.randn(features.shape[0], 1).astype(np.float32)

    assert target.shape == (4, 1)
    np.testing.assert_allclose(target, expected_target)
