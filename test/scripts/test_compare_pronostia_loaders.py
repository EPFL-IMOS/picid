"""Tests for the PRONOSTIA loader comparison script."""

from __future__ import annotations

from types import SimpleNamespace
import sys

import awkward as ak
import numpy as np

from scripts import compare_pronostia_loaders


def _make_unit(features_value: float, target_value: float):
    features = np.full((1, 2, 2), features_value, dtype=np.float32)
    target = np.full((1, 2, 1), target_value, dtype=np.float32)
    return ak.Array(features), ak.Array(target)


def _make_container(split_to_units: dict[str, list[dict]]) -> SimpleNamespace:
    features: dict[str, list] = {}
    target: dict[str, list] = {}
    unit_metadata: dict[str, list[dict]] = {}

    for split, units in split_to_units.items():
        features[split] = []
        target[split] = []
        unit_metadata[split] = []
        for unit in units:
            unit_features, unit_target = _make_unit(
                unit["features_value"], unit["target_value"]
            )
            features[split].append(unit_features)
            target[split].append(unit_target)
            unit_metadata[split].append(unit["metadata"])

    return SimpleNamespace(
        features=features,
        target=target,
        unit_metadata=unit_metadata,
    )


def test_compare_pronostia_loaders_prints_notebook_style_tables(
    monkeypatch, capsys, tmp_path
) -> None:
    phmd_container = _make_container(
        {
            "train": [
                {
                    "features_value": 1.0,
                    "target_value": 8690.0,
                    "metadata": {
                        "unit_name": "Bearing 1_2",
                        "dataset_subset": "Learning_set",
                    },
                }
            ],
            "val": [],
            "test": [
                {
                    "features_value": 2.0,
                    "target_value": 23730.0,
                    "metadata": {
                        "unit_name": "Bearing 1_3",
                        "dataset_subset": "Full_Test_Set",
                    },
                }
            ],
        }
    )
    raw_container = _make_container(
        {
            "train": [
                {
                    "features_value": 1.0,
                    "target_value": 8690.0,
                    "metadata": {
                        "unit_name": "Bearing 1_2",
                        "dataset_subset": "Learning_set",
                    },
                },
                {
                    "features_value": 2.0,
                    "target_value": 23730.0,
                    "metadata": {
                        "unit_name": "Bearing 1_3",
                        "dataset_subset": "Full_Test_Set",
                    },
                },
            ],
            "val": [],
            "test": [],
        }
    )

    class FakePhmdLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_data(self) -> None:
            return None

        def get_data(self):
            return phmd_container

    class FakeRawLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_data(self) -> None:
            return None

        def get_data(self):
            return raw_container

    monkeypatch.setattr(compare_pronostia_loaders, "PronostiaLoader", FakePhmdLoader)
    monkeypatch.setattr(
        compare_pronostia_loaders,
        "PronostiaIEEE2012Loader",
        FakeRawLoader,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_pronostia_loaders.py",
            "--raw-data-dir",
            str(tmp_path / "raw"),
            "--phmd-cache-dir",
            str(tmp_path / "phmd"),
            "--phmd-payload-cache",
            str(tmp_path / "payload.joblib"),
            "--no-download-raw",
        ],
    )

    exit_code = compare_pronostia_loaders.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "=== Raw PRONOSTIA IEEE 2012 split statistics ===" in captured.out
    assert "=== PHMD split membership ===" in captured.out
    assert "=== Raw IEEE 2012 in-domain split membership ===" in captured.out
    assert "=== PHMD vs raw IEEE 2012 comparison ===" in captured.out
    assert "=== Expected-behavior summary ===" in captured.out
    assert "Bearing 1_2" in captured.out
    assert "Bearing 1_3" in captured.out
