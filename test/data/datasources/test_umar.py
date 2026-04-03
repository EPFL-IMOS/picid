"""Tests for UMAR datasource loaders (room, building, load) and multisource merge."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from picid.data.datasources.base.exceptions import DatasourceStateError
from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.data.datasources.umar import (
    UMAR_BUILDING_ROOM_ORDER,
    UMAR_LOCAL_FEATURE_SCHEMA,
    UMAR_LOAD_FEATURE_COLUMNS,
    UMARBuildingLoader,
    UMARLoadLoader,
    UMARLoader,
)
from picid.data.preprocessing import TimeSplitter

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def _collate_key_value_batch():
    """Load collate without importing ``picid.data.datasets`` (avoids lightning deps in CI)."""
    root = Path(__file__).resolve().parents[3]
    path = root / "picid" / "data" / "datasets" / "collate_functions.py"
    spec = importlib.util.spec_from_file_location("collate_functions_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.collate_key_value_batch


EXPECTED_UMAR_LOAD_SENSOR_COLUMNS = [
    "R272__Flow",
    "R272__Occupancy",
    "R272__Setpoint_Temperature",
    "R272__Shade",
    "R272__Window",
    "R273__Flow",
    "R273__Occupancy",
    "R273__Setpoint_Temperature",
    "R273__Shade",
    "R273__Window",
    "R274__Flow",
    "R274__Occupancy",
    "R274__Setpoint_Temperature",
    "R274__Shade",
    "R274__Window",
    "R275__Flow",
    "R275__Setpoint_Temperature",
    "R276__Flow",
    "R276__Setpoint_Temperature",
]


@pytest.fixture
def umar_kwargs():
    return {
        "data_name": "UMAR_R272",
        "data_path": "/fake/path/inputs.pkl",
        "target_path": "/fake/path/targets.pkl",
        "room_id": "R272",
        "task_mode": "multivariate",
    }


@pytest.fixture
def umar_room_time_splitter() -> TimeSplitter:
    return TimeSplitter(
        train=0.7,
        val=0.1,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=[
            "features_local",
            "features_global",
            "features_local_schema_mask",
            "features_static",
            "target",
            "timestamps",
        ],
    )


def _mask_row(data_dict):
    return data_dict["features_local_schema_mask"][0]


class TestUMARRoomSchema:
    """Canonical local/global tensors and masks per room."""

    def test_load_data_emits_canonical_room_schema(self, umar_kwargs, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoader(**umar_kwargs)

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            data_dict = loader._load_data()

        assert data_dict["features_local"].shape == (100, 8)
        assert data_dict["features_local_schema_mask"].shape == (100, 8)
        assert data_dict["features_global"].shape == (100, 11)
        assert data_dict["features_static"].shape == (100, 1)
        assert data_dict["target"].shape == (100, 1)

        np.testing.assert_array_equal(
            _mask_row(data_dict),
            np.array([True, True, True, True, False, False, True, False], dtype=bool),
        )
        np.testing.assert_allclose(data_dict["features_local"][:, 0], 10.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 3], 13.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 6], 14.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 4], 0.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 5], 0.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 7], 0.0)
        np.testing.assert_array_equal(
            data_dict["timestamps"], df_inputs.index.to_numpy()
        )

        assert data_dict["metadata"]["unit_name"] == "R272"
        assert data_dict["metadata"]["column_map"]["features_local"] == list(
            UMAR_LOCAL_FEATURE_SCHEMA
        )
        assert data_dict["metadata"]["local_active_feature_names"] == [
            "Flow",
            "Occupancy",
            "Setpoint_Temperature",
            "Shade1",
            "Window1",
        ]

    def test_preserves_distinct_r273_shades_and_windows(self, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoader(
            data_name="UMAR_R273",
            data_path="/fake/path/inputs.pkl",
            target_path="/fake/path/targets.pkl",
            room_id="R273",
            task_mode="multivariate",
        )

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            data_dict = loader._load_data()

        np.testing.assert_array_equal(
            _mask_row(data_dict),
            np.ones(8, dtype=bool),
        )
        np.testing.assert_allclose(data_dict["features_local"][:, 3], 23.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 4], 24.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 5], 25.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 6], 26.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 7], 27.0)

    def test_zero_pads_missing_room_slots(self, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoader(
            data_name="UMAR_R275",
            data_path="/fake/path/inputs.pkl",
            target_path="/fake/path/targets.pkl",
            room_id="R275",
            task_mode="multivariate",
        )

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            data_dict = loader._load_data()

        np.testing.assert_array_equal(
            _mask_row(data_dict),
            np.array(
                [True, False, True, False, False, False, False, False], dtype=bool
            ),
        )
        np.testing.assert_allclose(data_dict["features_local"][:, 0], 40.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 2], 41.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 1], 0.0)
        np.testing.assert_allclose(data_dict["features_local"][:, 3:], 0.0)

    def test_keeps_canonical_schema_when_local_feature_bases_are_passed(
        self,
        mock_umar_data,
    ):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoader(
            data_name="UMAR_R272",
            data_path="/fake/path/inputs.pkl",
            target_path="/fake/path/targets.pkl",
            room_id="R272",
            task_mode="multivariate",
            local_feature_bases=[
                "Flow",
                "Occupancy",
                "Setpoint_Temperature",
                "Shade",
                "Window",
            ],
        )

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            data_dict = loader._load_data()

        assert loader.local_feature_schema == UMAR_LOCAL_FEATURE_SCHEMA
        assert data_dict["metadata"]["local_feature_schema"] == list(
            UMAR_LOCAL_FEATURE_SCHEMA
        )
        assert data_dict["features_local"].shape == (100, 8)
        np.testing.assert_array_equal(
            _mask_row(data_dict),
            np.array([True, True, True, True, False, False, True, False], dtype=bool),
        )


class TestSplitGetDataContract:
    """``load_data`` → ``split_data`` → ``get_data`` preserves masks and metadata."""

    def test_preserves_masks_and_column_map_through_split_get_data(
        self,
        umar_kwargs,
        umar_room_time_splitter: TimeSplitter,
        mock_umar_data,
    ):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoader(
            **umar_kwargs,
            data_splitter=umar_room_time_splitter,
        )

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            loader.load_data()
            loader.split_data()
            data = loader.get_data()
            split_dict = data.to_split_dict()

        expected_mask = np.array(
            [True, True, True, True, False, False, True, False],
            dtype=bool,
        )
        train_mask_block = data["features_local_schema_mask"]["train"][0]
        np.testing.assert_array_equal(train_mask_block[0], expected_mask)
        np.testing.assert_array_equal(
            split_dict["train"]["features_local_schema_mask"][0][0],
            expected_mask,
        )
        assert data.metadata["column_map"]["features_local"] == list(
            UMAR_LOCAL_FEATURE_SCHEMA
        )
        assert data.metadata["local_slot_sources"] == {
            "Flow": "R272_Flow",
            "Occupancy": "R272_Occupancy",
            "Setpoint_Temperature": "R272_Setpoint_Temperature",
            "Shade1": "R272_Shade",
            "Window1": "R272_Window",
        }

    def test_get_data_before_load_raises(self, umar_kwargs):
        loader = UMARLoader(**umar_kwargs)
        with pytest.raises(DatasourceStateError, match="must be loaded"):
            loader.get_data()


class TestMissingColumns:
    """Required columns must be present or load fails loudly."""

    def test_missing_target_raises_by_default(self, umar_kwargs, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoader(**umar_kwargs)
        df_targets = df_targets.drop(columns=["R272_Air_Temperature [C]"])

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            with pytest.raises(KeyError, match="Target column"):
                loader._load_data()

    def test_missing_global_feature_raises(self, umar_kwargs, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoader(**umar_kwargs)
        df_inputs = df_inputs.drop(columns=["Wind_Speed"])

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            with pytest.raises(KeyError, match="UMAR global feature columns missing"):
                loader._load_data()


class TestMultisourceHydra:
    """Multi-room composition and Hydra config wiring."""

    def test_multisource_merge_preserves_per_room_schema_masks(self, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        rooms = ["R272", "R273", "R274", "R275", "R276"]
        loader_args = {
            "data_name": "umar",
            "task_mode": "multivariate",
            "data_path": "/fake/in.pkl",
            "target_path": "/fake/tg.pkl",
            "timestamp_name": "datetime",
            "data_splitter": {
                "_target_": "picid.data.split_strategies.time_splitter.TimeSplitter",
                "train": 0.7,
                "val": 0.1,
                "test": None,
                "seq_len": 10,
                "pred_len": 5,
                "create_splits_for": [
                    "features_local",
                    "features_global",
                    "features_local_schema_mask",
                    "features_static",
                    "target",
                    "timestamps",
                ],
            },
        }
        cfg = OmegaConf.create(
            {
                "_target_": "picid.data.datasources.base.multi_source_loader.MultiSourceLoader",
                "_recursive_": False,
                "data_name": "umar",
                "task_mode": "multivariate",
                "multisource_data_splitter": None,
                "source_list": {r: r for r in rooms},
                **{
                    r: {
                        "_target_": "picid.data.datasources.umar.UMARLoader",
                        "room_id": r,
                        **loader_args,
                    }
                    for r in rooms
                },
            }
        )
        datasource = instantiate(cfg)

        with (
            patch("builtins.open"),
            patch(
                "pickle.load",
                side_effect=[df_inputs, df_targets] * 5,
            ),
        ):
            datasource.load_data()
            datasource.split_data()
            data = datasource.get_data()
            split_dict = data.to_split_dict()

        expected_room_order = ["R272", "R273", "R274", "R275", "R276"]
        expected_masks = {
            "R272": np.array(
                [True, True, True, True, False, False, True, False], dtype=bool
            ),
            "R273": np.ones(8, dtype=bool),
            "R274": np.array(
                [True, True, True, True, False, False, True, False], dtype=bool
            ),
            "R275": np.array(
                [True, False, True, False, False, False, False, False], dtype=bool
            ),
            "R276": np.array(
                [True, False, True, False, False, False, False, False], dtype=bool
            ),
        }

        assert len(data["features_local"]["train"]) == 5
        assert "features_local_schema_mask" in split_dict["train"]

        for split_name in ("train", "val", "test"):
            for idx, room_id in enumerate(expected_room_order):
                block = data["features_local_schema_mask"][split_name][idx]
                np.testing.assert_array_equal(block[0], expected_masks[room_id])
                assert np.all(block == block[0])

    def test_multisource_yaml_config_instantiates(self, mock_umar_data):
        """configs/datasource/umar.yaml composes with run.yaml when task lengths are set."""
        df_inputs, df_targets = mock_umar_data
        GlobalHydra.instance().clear()
        with hydra.initialize_config_dir(
            version_base="1.3", config_dir=str(CONFIGS_DIR)
        ):
            cfg = hydra.compose(
                config_name="run",
                overrides=[
                    "datasource=umar",
                    "experiment=umar_test",
                    "paths.data_dir=/tmp",
                    "hydra/hydra_logging=default",
                    "hydra/job_logging=default",
                ],
            )
            ds = instantiate(cfg.datasource)
            assert isinstance(ds, MultiSourceLoader)
            assert ds.get_source_names() == ("R272", "R273", "R274", "R275", "R276")

            with (
                patch("builtins.open"),
                patch(
                    "pickle.load",
                    side_effect=[df_inputs, df_targets] * 5,
                ),
            ):
                ds.load_data()
                ds.split_data()
                out = ds.get_data()
                assert len(out["features_local"]["train"]) == 5


class TestUMARLoadLoader:
    """Building-load (scenario 1) feature contract and split metadata."""

    def test_extracts_scenario1_contract(self, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoadLoader(
            data_name="UMAR_LOAD",
            data_path="/fake/path/inputs.pkl",
            target_path="/fake/path/targets.pkl",
            task_mode="univariate",
        )

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            data_dict = loader._load_data()

        assert list(data_dict["features"].columns) == list(UMAR_LOAD_FEATURE_COLUMNS)
        assert (
            list(data_dict["sensor_features"].columns)
            == EXPECTED_UMAR_LOAD_SENSOR_COLUMNS
        )
        assert list(data_dict["target"].columns) == ["Electric_Energy_Consumption [kW]"]
        assert isinstance(data_dict["timestamps"], pd.Series)
        assert data_dict["timestamps"].iloc[0] == df_inputs.index[0]
        assert data_dict["timestamps"].iloc[-1] == df_inputs.index[-1]
        assert (
            data_dict["metadata"]["target_name"] == "Electric_Energy_Consumption [kW]"
        )
        assert data_dict["metadata"]["feature_names"] == list(UMAR_LOAD_FEATURE_COLUMNS)
        assert data_dict["metadata"]["resolution"] == "15min"
        assert data_dict["metadata"]["column_map"] == {
            "features": list(UMAR_LOAD_FEATURE_COLUMNS),
            "sensor_features": EXPECTED_UMAR_LOAD_SENSOR_COLUMNS,
            "target": ["Electric_Energy_Consumption [kW]"],
        }
        assert "AC_mode" not in data_dict["features"].columns
        assert "DistrictCooling_Flow" not in data_dict["features"].columns
        assert "R272_Flow" not in data_dict["features"].columns
        np.testing.assert_allclose(data_dict["sensor_features"]["R272__Flow"], 10.0)
        np.testing.assert_allclose(data_dict["sensor_features"]["R273__Shade"], 24.0)
        np.testing.assert_allclose(data_dict["sensor_features"]["R273__Window"], 26.5)
        np.testing.assert_allclose(data_dict["sensor_features"]["R274__Window"], 34.0)
        assert "R275__Occupancy" not in data_dict["sensor_features"].columns
        assert "R276__Shade" not in data_dict["sensor_features"].columns

    def test_preserves_sensor_feature_column_map_through_split_get_data(
        self,
        mock_umar_data,
    ):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoadLoader(
            data_name="UMAR_LOAD",
            data_path="/fake/path/inputs.pkl",
            target_path="/fake/path/targets.pkl",
            task_mode="univariate",
            data_splitter=TimeSplitter(
                train=0.7,
                val=0.1,
                test=None,
                seq_len=10,
                pred_len=5,
                create_splits_for=[
                    "features",
                    "sensor_features",
                    "target",
                    "timestamps",
                ],
            ),
        )

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            loader.load_data()
            loader.split_data()
            data = loader.get_data()

        assert data.metadata["column_map"]["features"] == list(
            UMAR_LOAD_FEATURE_COLUMNS
        )
        assert (
            data.metadata["column_map"]["sensor_features"]
            == EXPECTED_UMAR_LOAD_SENSOR_COLUMNS
        )
        assert (
            list(data["sensor_features"]["train"][0].columns)
            == EXPECTED_UMAR_LOAD_SENSOR_COLUMNS
        )
        assert (
            list(data["sensor_features"]["val"][0].columns)
            == EXPECTED_UMAR_LOAD_SENSOR_COLUMNS
        )
        assert (
            list(data["sensor_features"]["test"][0].columns)
            == EXPECTED_UMAR_LOAD_SENSOR_COLUMNS
        )

    def test_missing_target_raises(self, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoadLoader(
            data_name="UMAR_LOAD",
            data_path="/fake/path/inputs.pkl",
            target_path="/fake/path/targets.pkl",
            task_mode="univariate",
        )
        df_targets = df_targets.drop(columns=["Electric_Energy_Consumption [kW]"])

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            with pytest.raises(KeyError, match="Electric_Energy_Consumption"):
                loader._load_data()

    def test_missing_feature_raises(self, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        loader = UMARLoadLoader(
            data_name="UMAR_LOAD",
            data_path="/fake/path/inputs.pkl",
            target_path="/fake/path/targets.pkl",
            task_mode="univariate",
        )
        df_inputs = df_inputs.drop(columns=["Wind_Speed"])

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            with pytest.raises(KeyError, match="UMAR load feature columns missing"):
                loader._load_data()


class TestUMARBuildingLoader:
    """Multi-room building tensor layout and stable room order."""

    def test_uses_stable_room_order(self, mock_umar_data):
        df_inputs, df_targets = mock_umar_data
        loader = UMARBuildingLoader(
            data_name="UMAR_BUILDING",
            data_path="/fake/path/inputs.pkl",
            target_path="/fake/path/targets.pkl",
            task_mode="multivariate",
            room_ids=["R272", "R273", "R274", "R275", "R276"],
        )

        with (
            patch("builtins.open"),
            patch("pickle.load", side_effect=[df_inputs, df_targets]),
        ):
            data_dict = loader._load_data()

        assert data_dict["features_local"].shape == (100, 25)
        assert data_dict["features_global"].shape == (100, 11)
        assert data_dict["target"].shape == (100, 5)
        assert list(data_dict["metadata"]["room_ids"]) == list(UMAR_BUILDING_ROOM_ORDER)
        assert data_dict["metadata"]["target_names"] == [
            "R272",
            "R273",
            "R274",
            "R275",
            "R276",
        ]
        np.testing.assert_array_equal(
            data_dict["timestamps"], df_inputs.index.to_numpy()
        )
        np.testing.assert_allclose(
            data_dict["target"][:, 0],
            df_targets["R272_Air_Temperature [C]"].to_numpy(),
        )
        np.testing.assert_allclose(
            data_dict["target"][:, 4],
            df_targets["R276_Air_Temperature [C]"].to_numpy(),
        )


class TestCollateHelper:
    """Standalone collate path used by forecasting-style batches."""

    def test_collate_key_value_batch_accepts_numpy_arrays(self):
        collate_key_value_batch = _collate_key_value_batch()
        batch = [
            {
                "features": np.array([[[1.0, 2.0]]], dtype=np.float32),
                "target": np.array([[[3.0]]], dtype=np.float32),
            },
            {
                "features": np.array([[[4.0, 5.0]]], dtype=np.float32),
                "target": np.array([[[6.0]]], dtype=np.float32),
            },
        ]
        out = collate_key_value_batch(batch)
        assert torch.is_tensor(out["features"])
        assert out["features"].shape == (2, 1, 2)
        assert out["target"].shape == (2, 1, 1)
