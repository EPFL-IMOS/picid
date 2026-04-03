"""Contracts for shared datasource test layout builders (canonical columns / keys)."""

from __future__ import annotations

from test.fixtures.datasource_layouts import (
    build_mock_umar_multivariate_pickles_dataframes,
    make_threew_frame,
)


def test_make_threew_frame_has_expected_columns():
    frame = make_threew_frame(class_id=3, n_rows=4)
    assert {"timestamp", "QGL", "class"} <= set(frame.columns)


def test_build_mock_umar_pickles_has_expected_canonical_columns():
    df_inputs, df_targets = build_mock_umar_multivariate_pickles_dataframes()
    assert str(df_inputs.index.name).lower() == "datetime"
    assert str(df_targets.index.name).lower() == "datetime"
    assert len(df_inputs) == len(df_targets) == 100
    assert {
        "AC_mode",
        "R272_Flow",
        "R273_Shade1",
        "R276_Setpoint_Temperature",
    } <= set(df_inputs.columns)
    assert {
        "Electric_Energy_Consumption [kW]",
        "R272_Air_Temperature [C]",
    } <= set(df_targets.columns)
