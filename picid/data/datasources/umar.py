"""UMAR (Urban Mining and Recycling) building / room datasource loaders."""

import logging
import pickle

import numpy as np
import pandas as pd

from picid.data.datasources.base.single_source_loader import SingleSourceLoader

logger = logging.getLogger(__name__)

UMAR_BUILDING_ROOM_ORDER = ("R272", "R273", "R274", "R275", "R276")
UMAR_LOCAL_FEATURE_SCHEMA = (
    "Flow",
    "Occupancy",
    "Setpoint_Temperature",
    "Shade1",
    "Shade2",
    "Shade3",
    "Window1",
    "Window2",
)
UMAR_ROOM_LOCAL_SLOT_SOURCE_SUFFIXES = {
    "R272": {
        "Flow": "Flow",
        "Occupancy": "Occupancy",
        "Setpoint_Temperature": "Setpoint_Temperature",
        "Shade1": "Shade",
        "Window1": "Window",
    },
    "R273": {
        "Flow": "Flow",
        "Occupancy": "Occupancy",
        "Setpoint_Temperature": "Setpoint_Temperature",
        "Shade1": "Shade1",
        "Shade2": "Shade2",
        "Shade3": "Shade3",
        "Window1": "Window1",
        "Window2": "Window2",
    },
    "R274": {
        "Flow": "Flow",
        "Occupancy": "Occupancy",
        "Setpoint_Temperature": "Setpoint_Temperature",
        "Shade1": "Shade",
        "Window1": "Window",
    },
    "R275": {
        "Flow": "Flow",
        "Setpoint_Temperature": "Setpoint_Temperature",
    },
    "R276": {
        "Flow": "Flow",
        "Setpoint_Temperature": "Setpoint_Temperature",
    },
}

UMAR_LOAD_FEATURE_COLUMNS = (
    "DewPoint_Temperature",
    "Diffuse_SolarRadiation",
    "Direct_SolarRadiation",
    "DryBulb_Temperature",
    "Relative_Humidity",
    "Wind_Direction",
    "Wind_Speed",
)
UMAR_BUILDING_SENSOR_BASES = (
    "Flow",
    "Occupancy",
    "Setpoint_Temperature",
    "Shade",
    "Window",
)


class UMARLoader(SingleSourceLoader):
    """Loader for one UMAR room: building-level features plus a fixed local feature schema."""

    DEFAULT_GLOBAL_FEATURES = (
        "AC_mode",
        "DewPoint_Temperature",
        "Diffuse_SolarRadiation",
        "Direct_SolarRadiation",
        "DistrictCooling_Flow",
        "DistrictHeating_Flow",
        "District_Network_Temperature",
        "DryBulb_Temperature",
        "Relative_Humidity",
        "Wind_Direction",
        "Wind_Speed",
    )
    DEFAULT_LOCAL_FEATURE_SCHEMA = UMAR_LOCAL_FEATURE_SCHEMA
    DEFAULT_LOCAL_FEATURE_BASES = (
        "Flow",
        "Occupancy",
        "Setpoint_Temperature",
        "Shade",
        "Window",
    )
    DEFAULT_ROOM_ORDER = UMAR_BUILDING_ROOM_ORDER
    ROOM_LOCAL_SLOT_SOURCE_SUFFIXES = UMAR_ROOM_LOCAL_SLOT_SOURCE_SUFFIXES
    STATIC_FEATURE_NAMES = ("room_id_numeric",)

    def __init__(
        self,
        data_path: str,
        target_path: str,
        room_id: str,
        timestamp_name: str = "datetime",
        global_feature_columns: list[str] | tuple[str, ...] | None = None,
        local_feature_bases: list[str] | tuple[str, ...] | None = None,
        missing_local_value: float = 0.0,
        missing_target_policy: str = "raise",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.target_path = target_path
        self.room_id = room_id
        self.timestamp_name = timestamp_name
        self.global_feature_columns = tuple(
            global_feature_columns or self.DEFAULT_GLOBAL_FEATURES
        )
        self.local_feature_bases = tuple(
            local_feature_bases or self.DEFAULT_LOCAL_FEATURE_BASES
        )
        self.local_feature_schema = tuple(self.DEFAULT_LOCAL_FEATURE_SCHEMA)
        self.missing_local_value = missing_local_value
        if missing_target_policy not in {"raise", "zeros"}:
            raise ValueError(
                "missing_target_policy must be either 'raise' or 'zeros', got "
                f"{missing_target_policy!r}."
            )
        self.missing_target_policy = missing_target_policy

    def _extract_timestamps(self, df_inputs: pd.DataFrame) -> np.ndarray:
        if isinstance(df_inputs.index, pd.DatetimeIndex):
            return df_inputs.index.to_numpy()

        if self.timestamp_name in df_inputs.columns:
            return pd.to_datetime(df_inputs[self.timestamp_name]).to_numpy()
        return np.arange(len(df_inputs))

    def _extract_global_features(self, df_inputs: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [
            column
            for column in self.global_feature_columns
            if column not in df_inputs.columns
        ]
        if missing_columns:
            raise KeyError(
                "UMAR global feature columns missing from inputs: "
                f"{missing_columns}."
            )
        return df_inputs.loc[:, self.global_feature_columns].copy()

    def _resolve_room_local_slot_source_suffix(
        self, room_id: str, slot_name: str
    ) -> str:
        return self.ROOM_LOCAL_SLOT_SOURCE_SUFFIXES.get(room_id, {}).get(
            slot_name, slot_name
        )

    def _extract_local_features_and_metadata(
        self, df_inputs: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
        local_df = pd.DataFrame(index=df_inputs.index)
        schema_mask = np.zeros(len(self.local_feature_schema), dtype=bool)
        slot_sources: dict[str, str] = {}

        for idx, slot_name in enumerate(self.local_feature_schema):
            raw_suffix = self._resolve_room_local_slot_source_suffix(
                room_id=self.room_id,
                slot_name=slot_name,
            )
            col_name = f"{self.room_id}_{raw_suffix}"
            if col_name in df_inputs.columns:
                local_df[slot_name] = df_inputs[col_name].to_numpy()
                schema_mask[idx] = True
                slot_sources[slot_name] = col_name
            else:
                local_df[slot_name] = np.full(
                    len(df_inputs),
                    self.missing_local_value,
                    dtype=np.float32,
                )

        return local_df, schema_mask, slot_sources

    def _extract_local_features(self, df_inputs: pd.DataFrame) -> pd.DataFrame:
        local_df = pd.DataFrame(index=df_inputs.index)

        for base in self.local_feature_bases:
            new_col_name = f"Room_{base}"
            local_df[new_col_name] = self._extract_room_local_series(
                df_inputs=df_inputs,
                room_id=self.room_id,
                base=base,
            )

        return local_df

    def _extract_target(self, df_targets: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Room_Air_Temperature": self._extract_room_target_series(
                    df_targets=df_targets,
                    room_id=self.room_id,
                )
            },
            index=df_targets.index,
        )

    def _build_metadata(
        self,
        local_df: pd.DataFrame,
        global_df: pd.DataFrame,
        target_df: pd.DataFrame,
        local_schema_mask: np.ndarray,
        local_slot_sources: dict[str, str],
    ) -> dict:
        active_local_feature_names = [
            feature_name
            for feature_name, is_active in zip(
                self.local_feature_schema, local_schema_mask, strict=True
            )
            if is_active
        ]
        return {
            "unit_name": self.room_id,
            "unit_id": self.room_id,
            "local_feature_schema": list(self.local_feature_schema),
            "local_active_feature_names": active_local_feature_names,
            "local_slot_sources": dict(local_slot_sources),
            "column_map": {
                "features_local": list(local_df.columns),
                "features_global": list(global_df.columns),
                "features_static": list(self.STATIC_FEATURE_NAMES),
                "target": list(target_df.columns),
            },
        }

    def _extract_room_local_series(
        self,
        df_inputs: pd.DataFrame,
        room_id: str,
        base: str,
    ) -> np.ndarray:
        col_name = f"{room_id}_{base}"

        if col_name in df_inputs.columns:
            return df_inputs[col_name].to_numpy()

        matching_cols = [
            column
            for column in df_inputs.columns
            if column.startswith(col_name) and column[len(col_name) :].isdigit()
        ]
        if matching_cols:
            return df_inputs[matching_cols].mean(axis=1).to_numpy()

        return np.full(len(df_inputs), self.missing_local_value, dtype=np.float32)

    def _extract_room_target_series(
        self,
        df_targets: pd.DataFrame,
        room_id: str,
    ) -> np.ndarray:
        target_col = f"{room_id}_Air_Temperature [C]"
        if target_col in df_targets.columns:
            return df_targets[target_col].to_numpy()

        if self.missing_target_policy == "zeros":
            logger.warning("Target column %s missing in target file.", target_col)
            return np.zeros(len(df_targets), dtype=np.float32)

        raise KeyError(
            f"Target column {target_col!r} missing in UMAR target file for room "
            f"{room_id!r}."
        )

    def _extract_static_features(self) -> np.ndarray:
        room_numeric_part = self.room_id.removeprefix("R")
        if room_numeric_part.isdigit():
            return np.array([float(room_numeric_part)], dtype=np.float32)
        return np.array([1.0], dtype=np.float32)

    def _validate_alignment(
        self, df_inputs: pd.DataFrame, target_df: pd.DataFrame
    ) -> None:
        if len(df_inputs) != len(target_df):
            raise ValueError(
                f"UMAR input/target length mismatch for room {self.room_id!r}: "
                f"{len(df_inputs)} != {len(target_df)}."
            )

        if not df_inputs.index.equals(target_df.index):
            raise ValueError(
                f"UMAR input/target timestamps are misaligned for room {self.room_id!r}."
            )

    def _broadcast_row(self, row: np.ndarray, n_rows: int) -> np.ndarray:
        """Repeat a row vector along time so TimeSplitter can slice it."""
        return np.broadcast_to(row, (n_rows, row.shape[-1])).copy()

    def _load_data(self) -> dict:
        with open(self.data_path, "rb") as f:
            df_inputs = pickle.load(f)

        with open(self.target_path, "rb") as f:
            df_targets = pickle.load(f)

        timestamps = self._extract_timestamps(df_inputs)
        global_df = self._extract_global_features(df_inputs)
        local_df, local_schema_mask, local_slot_sources = (
            self._extract_local_features_and_metadata(df_inputs)
        )
        target_df = self._extract_target(df_targets)
        static_features = self._extract_static_features()

        self._validate_alignment(df_inputs=df_inputs, target_df=target_df)

        n = len(timestamps)
        mask_row = local_schema_mask.astype(np.bool_)[np.newaxis, :]
        static_row = static_features.astype(np.float32)[np.newaxis, :]

        return {
            "features_local": local_df.values.astype(np.float32),
            "features_local_schema_mask": self._broadcast_row(mask_row, n),
            "features_global": global_df.values.astype(np.float32),
            "features_static": self._broadcast_row(static_row, n),
            "target": target_df.values.astype(np.float32),
            "timestamps": timestamps,
            "metadata": self._build_metadata(
                local_df=local_df,
                global_df=global_df,
                target_df=target_df,
                local_schema_mask=local_schema_mask,
                local_slot_sources=local_slot_sources,
            ),
        }


class UMARLoadLoader(SingleSourceLoader):
    """Building-level UMAR electric load forecasting (weather + per-room sensors)."""

    DEFAULT_FEATURE_COLUMNS = UMAR_LOAD_FEATURE_COLUMNS
    DEFAULT_SENSOR_ROOM_ORDER = UMAR_BUILDING_ROOM_ORDER
    DEFAULT_SENSOR_FEATURE_BASES = UMAR_BUILDING_SENSOR_BASES

    def __init__(
        self,
        data_path: str,
        target_path: str,
        target_name: str = "Electric_Energy_Consumption [kW]",
        feature_columns: list[str] | tuple[str, ...] | None = None,
        timestamp_name: str = "datetime",
        building_id: str = "UMAR_BUILDING",
        resolution: str = "15min",
        sensor_room_order: list[str] | tuple[str, ...] | None = None,
        sensor_feature_bases: list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.target_path = target_path
        self.target_name = target_name
        self.feature_columns = tuple(feature_columns or self.DEFAULT_FEATURE_COLUMNS)
        self.timestamp_name = timestamp_name
        self.building_id = building_id
        self.resolution = resolution
        self.sensor_room_order = tuple(
            sensor_room_order or self.DEFAULT_SENSOR_ROOM_ORDER
        )
        self.sensor_feature_bases = tuple(
            sensor_feature_bases or self.DEFAULT_SENSOR_FEATURE_BASES
        )

    def _extract_timestamps(self, df_inputs: pd.DataFrame) -> pd.Series:
        if isinstance(df_inputs.index, pd.DatetimeIndex):
            return pd.Series(df_inputs.index, name=self.timestamp_name)

        if self.timestamp_name in df_inputs.columns:
            return pd.Series(
                pd.to_datetime(df_inputs[self.timestamp_name]),
                name=self.timestamp_name,
            )
        raise ValueError(
            "UMAR load inputs must provide timestamps either via a DatetimeIndex "
            f"or a {self.timestamp_name!r} column."
        )

    def _extract_features(self, df_inputs: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [
            column for column in self.feature_columns if column not in df_inputs.columns
        ]
        if missing_columns:
            raise KeyError(
                "UMAR load feature columns missing from inputs: " f"{missing_columns}."
            )
        return df_inputs.loc[:, self.feature_columns].copy()

    @staticmethod
    def _extract_sensor_series(
        df_inputs: pd.DataFrame,
        room_id: str,
        base: str,
    ) -> np.ndarray | None:
        col_name = f"{room_id}_{base}"
        if col_name in df_inputs.columns:
            return df_inputs[col_name].to_numpy()

        matching_cols = sorted(
            column
            for column in df_inputs.columns
            if column.startswith(col_name) and column[len(col_name) :].isdigit()
        )
        if matching_cols:
            return df_inputs[matching_cols].mean(axis=1).to_numpy()

        return None

    def _extract_sensor_features(self, df_inputs: pd.DataFrame) -> pd.DataFrame:
        sensor_df = pd.DataFrame(index=df_inputs.index)
        for room_id in self.sensor_room_order:
            for base in self.sensor_feature_bases:
                sensor_series = self._extract_sensor_series(
                    df_inputs=df_inputs,
                    room_id=room_id,
                    base=base,
                )
                if sensor_series is None:
                    continue
                sensor_df[f"{room_id}__{base}"] = sensor_series
        return sensor_df

    def _extract_target(self, df_targets: pd.DataFrame) -> pd.DataFrame:
        if self.target_name not in df_targets.columns:
            raise KeyError(
                f"Target column {self.target_name!r} missing in UMAR target file."
            )
        return df_targets.loc[:, [self.target_name]].copy()

    def _validate_alignment(
        self,
        df_inputs: pd.DataFrame,
        target_df: pd.DataFrame,
        timestamps: pd.Series,
    ) -> None:
        if len(df_inputs) != len(target_df):
            raise ValueError(
                "UMAR load input/target length mismatch: "
                f"{len(df_inputs)} != {len(target_df)}."
            )
        if len(df_inputs) != len(timestamps):
            raise ValueError(
                "UMAR load input/timestamp length mismatch: "
                f"{len(df_inputs)} != {len(timestamps)}."
            )
        if not df_inputs.index.equals(target_df.index):
            raise ValueError("UMAR load input/target timestamps are misaligned.")

    def _load_data(self) -> dict:
        with open(self.data_path, "rb") as f:
            df_inputs = pickle.load(f)

        with open(self.target_path, "rb") as f:
            df_targets = pickle.load(f)

        timestamps = self._extract_timestamps(df_inputs)
        feature_df = self._extract_features(df_inputs)
        sensor_feature_df = self._extract_sensor_features(df_inputs)
        target_df = self._extract_target(df_targets)

        self._validate_alignment(
            df_inputs=df_inputs,
            target_df=target_df,
            timestamps=timestamps,
        )

        feature_names = list(feature_df.columns)
        sensor_feature_names = list(sensor_feature_df.columns)
        return {
            "features": feature_df,
            "sensor_features": sensor_feature_df,
            "target": target_df,
            "timestamps": timestamps,
            "metadata": {
                "unit_name": self.building_id,
                "unit_id": self.building_id,
                "target_name": self.target_name,
                "feature_names": feature_names,
                "resolution": self.resolution,
                "column_map": {
                    "features": feature_names,
                    "sensor_features": sensor_feature_names,
                    "target": [self.target_name],
                },
            },
        }


class UMARBuildingLoader(UMARLoader):
    """Multi-room building tensor layout with stable room ordering."""

    def __init__(
        self,
        data_path: str,
        target_path: str,
        room_ids: list[str] | tuple[str, ...] | None = None,
        timestamp_name: str = "datetime",
        global_feature_columns: list[str] | tuple[str, ...] | None = None,
        local_feature_bases: list[str] | tuple[str, ...] | None = None,
        missing_local_value: float = 0.0,
        missing_target_policy: str = "raise",
        building_id: str = "UMAR_BUILDING",
        **kwargs,
    ):
        resolved_room_ids = tuple(room_ids or self.DEFAULT_ROOM_ORDER)
        super().__init__(
            data_path=data_path,
            target_path=target_path,
            room_id=resolved_room_ids[0],
            timestamp_name=timestamp_name,
            global_feature_columns=global_feature_columns,
            local_feature_bases=local_feature_bases,
            missing_local_value=missing_local_value,
            missing_target_policy=missing_target_policy,
            **kwargs,
        )
        self.room_ids = resolved_room_ids
        self.building_id = building_id

    def _extract_local_features(self, df_inputs: pd.DataFrame) -> pd.DataFrame:
        local_df = pd.DataFrame(index=df_inputs.index)
        for room_id in self.room_ids:
            for base in self.local_feature_bases:
                local_df[f"{room_id}_{base}"] = self._extract_room_local_series(
                    df_inputs=df_inputs,
                    room_id=room_id,
                    base=base,
                )
        return local_df

    def _extract_target(self, df_targets: pd.DataFrame) -> pd.DataFrame:
        target_df = pd.DataFrame(index=df_targets.index)
        for room_id in self.room_ids:
            target_df[room_id] = self._extract_room_target_series(
                df_targets=df_targets,
                room_id=room_id,
            )
        return target_df

    def _extract_static_features(self) -> np.ndarray:
        raise NotImplementedError(
            "UMARBuildingLoader does not expose features_static. "
            "Use building-level metadata instead."
        )

    def _validate_alignment(
        self, df_inputs: pd.DataFrame, target_df: pd.DataFrame
    ) -> None:
        if len(df_inputs) != len(target_df):
            raise ValueError(
                "UMAR building input/target length mismatch: "
                f"{len(df_inputs)} != {len(target_df)}."
            )

        if not df_inputs.index.equals(target_df.index):
            raise ValueError("UMAR building input/target timestamps are misaligned.")

    def _load_data(self) -> dict:
        with open(self.data_path, "rb") as f:
            df_inputs = pickle.load(f)

        with open(self.target_path, "rb") as f:
            df_targets = pickle.load(f)

        timestamps = self._extract_timestamps(df_inputs)
        global_df = self._extract_global_features(df_inputs)
        local_df = self._extract_local_features(df_inputs)
        target_df = self._extract_target(df_targets)

        self._validate_alignment(df_inputs=df_inputs, target_df=target_df)

        return {
            "features_local": local_df.values.astype(np.float32),
            "features_global": global_df.values.astype(np.float32),
            "target": target_df.values.astype(np.float32),
            "timestamps": timestamps,
            "metadata": {
                "unit_name": self.building_id,
                "unit_id": self.building_id,
                "target_names": list(self.room_ids),
                "room_ids": list(self.room_ids),
                "local_feature_bases": list(self.local_feature_bases),
            },
        }
