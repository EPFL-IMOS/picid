"""Synthetic filesystem layouts and in-memory frames for datasource loader tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def touch_threew_instance(tmp_path: Path, class_id: int, instance_name: str) -> Path:
    """Create an empty parquet placeholder at ``dataset/<class_id>/<instance>.parquet``."""
    class_dir = tmp_path / "dataset" / str(class_id)
    class_dir.mkdir(parents=True, exist_ok=True)
    instance_path = class_dir / f"{instance_name}.parquet"
    instance_path.touch()
    return instance_path


def write_threew_folds(tmp_path: Path, rows: list[tuple[str, int, bool]]) -> None:
    """Write ``folds_clf_02.csv`` under ``dataset/folds/`` (``ThreeWLoader`` default)."""
    folds_dir = tmp_path / "dataset" / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["instancia", "fold", "is_ova"])
    df.to_csv(folds_dir / "folds_clf_02.csv", index=False)


def write_threew_folds_dataset_layout(
    tmp_path: Path, rows: list[tuple[str, int, bool]]
) -> None:
    """Alias for :func:`write_threew_folds` (explicit ``dataset/folds`` layout)."""
    write_threew_folds(tmp_path, rows)


def touch_threew_instance_dataset_layout(
    tmp_path: Path, class_id: int, instance_name: str
) -> Path:
    """Alias for :func:`touch_threew_instance` (explicit ``dataset/<class>`` layout)."""
    return touch_threew_instance(tmp_path, class_id, instance_name)


def make_threew_frame_with_class_series(
    class_ids: list[int] | np.ndarray,
) -> pd.DataFrame:
    """Synthetic 3W-like frame with a per-row ``class`` column."""
    series = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    n_rows = int(series.shape[0])
    values = np.arange(n_rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="min"),
            "P-PDG": values,
            "P-TPT": values + 1.0,
            "T-TPT": values + 2.0,
            "P-MON-CKP": values + 3.0,
            "T-JUS-CKP": values + 4.0,
            "P-JUS-CKGL": values + 5.0,
            "T-JUS-CKGL": values + 6.0,
            "QGL": values + 7.0,
            "class": series,
        }
    )


def make_threew_frame(class_id: int, n_rows: int = 6) -> pd.DataFrame:
    """Deterministic in-memory frame matching 3W sensor columns and constant ``class``."""
    values = np.arange(n_rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="min"),
            "P-PDG": values,
            "P-TPT": values + 1.0,
            "T-TPT": values + 2.0,
            "P-MON-CKP": values + 3.0,
            "T-JUS-CKP": values + 4.0,
            "P-JUS-CKGL": values + 5.0,
            "T-JUS-CKGL": values + 6.0,
            "QGL": values + 7.0,
            "class": np.full(n_rows, class_id, dtype=np.int64),
        }
    )


def build_mock_umar_multivariate_pickles_dataframes() -> (
    tuple[pd.DataFrame, pd.DataFrame]
):
    """Return (inputs, targets) DataFrames matching UMAR pickle tests (100 rows, 15min index)."""
    idx = pd.date_range("2019-01-01", periods=100, freq="15min")

    inputs_dict = {
        "AC_mode": np.ones(100, dtype=np.float32),
        "DewPoint_Temperature": np.linspace(0.0, 1.0, 100, dtype=np.float32),
        "Diffuse_SolarRadiation": np.linspace(1.0, 2.0, 100, dtype=np.float32),
        "Direct_SolarRadiation": np.linspace(2.0, 3.0, 100, dtype=np.float32),
        "DistrictCooling_Flow": np.linspace(3.0, 4.0, 100, dtype=np.float32),
        "DistrictHeating_Flow": np.linspace(4.0, 5.0, 100, dtype=np.float32),
        "District_Network_Temperature": np.linspace(5.0, 6.0, 100, dtype=np.float32),
        "DryBulb_Temperature": np.linspace(6.0, 7.0, 100, dtype=np.float32),
        "Relative_Humidity": np.linspace(7.0, 8.0, 100, dtype=np.float32),
        "Wind_Direction": np.linspace(8.0, 9.0, 100, dtype=np.float32),
        "Wind_Speed": np.linspace(9.0, 10.0, 100, dtype=np.float32),
        "R272_Flow": np.full(100, 10.0, dtype=np.float32),
        "R272_Occupancy": np.full(100, 11.0, dtype=np.float32),
        "R272_Setpoint_Temperature": np.full(100, 12.0, dtype=np.float32),
        "R272_Shade": np.full(100, 13.0, dtype=np.float32),
        "R272_Window": np.full(100, 14.0, dtype=np.float32),
        "R273_Flow": np.full(100, 20.0, dtype=np.float32),
        "R273_Occupancy": np.full(100, 21.0, dtype=np.float32),
        "R273_Setpoint_Temperature": np.full(100, 22.0, dtype=np.float32),
        "R273_Shade1": np.full(100, 23.0, dtype=np.float32),
        "R273_Shade2": np.full(100, 24.0, dtype=np.float32),
        "R273_Shade3": np.full(100, 25.0, dtype=np.float32),
        "R273_Window1": np.full(100, 26.0, dtype=np.float32),
        "R273_Window2": np.full(100, 27.0, dtype=np.float32),
        "R274_Flow": np.full(100, 30.0, dtype=np.float32),
        "R274_Occupancy": np.full(100, 31.0, dtype=np.float32),
        "R274_Setpoint_Temperature": np.full(100, 32.0, dtype=np.float32),
        "R274_Shade": np.full(100, 33.0, dtype=np.float32),
        "R274_Window": np.full(100, 34.0, dtype=np.float32),
        "R275_Flow": np.full(100, 40.0, dtype=np.float32),
        "R275_Setpoint_Temperature": np.full(100, 41.0, dtype=np.float32),
        "R276_Flow": np.full(100, 50.0, dtype=np.float32),
        "R276_Setpoint_Temperature": np.full(100, 51.0, dtype=np.float32),
    }
    df_inputs = pd.DataFrame(inputs_dict, index=idx)
    df_inputs.index.name = "Datetime"

    targets_dict = {
        "Electric_Energy_Consumption [kW]": np.linspace(
            100.0, 200.0, 100, dtype=np.float32
        ),
        "R272_Air_Temperature [C]": np.full(100, 18.0, dtype=np.float32),
        "R273_Air_Temperature [C]": np.full(100, 19.0, dtype=np.float32),
        "R274_Air_Temperature [C]": np.full(100, 20.0, dtype=np.float32),
        "R275_Air_Temperature [C]": np.full(100, 21.0, dtype=np.float32),
        "R276_Air_Temperature [C]": np.full(100, 22.0, dtype=np.float32),
    }
    df_targets = pd.DataFrame(targets_dict, index=idx)
    df_targets.index.name = "datetime"

    return df_inputs, df_targets
