from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from picid_report_dashboard.data import ResultsLoader, _safe_path_fragment


CANONICAL_LSTM = "forecasters.lstm_model.LSTM_Forecaster"
LEGACY_LSTM = "baselines.lstm_model.LSTM_Forecaster"


def _write_results(path, dataset: str, model: str, value: float) -> None:
    path.parent.mkdir(parents=True)
    xr.Dataset(
        {
            "mean": xr.DataArray(
                np.array([[[value]]]),
                dims=("dataset", "model", "metric_key"),
                coords={
                    "dataset": [dataset],
                    "model": [model],
                    "metric_key": ["test/mae"],
                },
            )
        }
    ).to_netcdf(path)


def test_legacy_lstm_uses_package_canonical_model_and_finds_hp_config(
    tmp_path,
) -> None:
    old_project = tmp_path / "old_project"
    moved_project = tmp_path / "padding_project"
    padding_dataset = "MultiSource_concepts_N-CMAPSS"
    _write_results(old_project / "results.nc", "Legacy dataset", LEGACY_LSTM, 1.0)
    _write_results(moved_project / "results.nc", padding_dataset, CANONICAL_LSTM, 2.0)

    stats_dir = old_project / "tables"
    stats_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Dataset": ["Legacy dataset"],
            "Model": [LEGACY_LSTM],
            "Total Runs": [10],
        }
    ).to_csv(stats_dir / "experiment_stats.csv", index=False)

    hp_dir = moved_project / "tables" / "hp_configs"
    hp_dir.mkdir(parents=True)
    hp_path = hp_dir / (
        f"{_safe_path_fragment(padding_dataset)}_{_safe_path_fragment(CANONICAL_LSTM)}.nc"
    )
    xr.Dataset(
        coords={
            "config": [0, 1],
            "task_definition.padding_left_flag": (
                "config",
                ["False", "True"],
            ),
        }
    ).to_netcdf(hp_path)

    loader = ResultsLoader(str(tmp_path))
    dashboard_dataset = f"{padding_dataset} [padding_project]"

    assert CANONICAL_LSTM in loader.models
    assert LEGACY_LSTM not in loader.models
    assert (
        loader.xarray_dataset["mean"]
        .sel(
            dataset=dashboard_dataset,
            model=CANONICAL_LSTM,
            metric_key="test/mae",
        )
        .item()
        == 2.0
    )
    assert (CANONICAL_LSTM, LEGACY_LSTM) in loader.applied_model_aliases

    stats = loader.stats_df
    assert stats is not None
    assert stats.loc[0, "Model"] == CANONICAL_LSTM

    hp_ds = loader.hp_impact_ds(dashboard_dataset, CANONICAL_LSTM)
    assert hp_ds is not None
    assert hp_ds.coords["task_definition.padding_left_flag"].values.tolist() == [
        "False",
        "True",
    ]
