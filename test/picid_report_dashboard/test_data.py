from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from picid_report_dashboard.data import ResultsLoader, _safe_path_fragment


CANONICAL_LSTM = "forecasters.lstm_model.LSTM_Forecaster"
LEGACY_LSTM = "baselines.lstm_model.LSTM_Forecaster"
WRAPPER_ALIASES = {
    "model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": (
        "estimators.cnn1d.wrapper.CNN1D_Wrapper"
    ),
    "model.wrappers.mlp_wrapper.MLPWrapper": "estimators.mlp.wrapper.MLPWrapper",
    "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": (
        "estimators.statistical.wrapper.StatisticalBaselineWrapper (exponential)"
    ),
    "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": (
        "estimators.statistical.wrapper.StatisticalBaselineWrapper (linear)"
    ),
}


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


def _write_multi_model_results(
    path,
    dataset: str,
    models: list[str],
    values: list[float],
) -> None:
    path.parent.mkdir(parents=True)
    xr.Dataset(
        {
            "mean": xr.DataArray(
                np.asarray(values, dtype=float).reshape(1, len(models), 1),
                dims=("dataset", "model", "metric_key"),
                coords={
                    "dataset": [dataset],
                    "model": models,
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


def test_wrapper_aliases_merge_models_and_find_legacy_hp_configs(tmp_path) -> None:
    dataset = "Dataset"
    legacy_project = tmp_path / "legacy_project"
    current_project = tmp_path / "current_project"
    legacy_models = list(WRAPPER_ALIASES)
    canonical_models = list(WRAPPER_ALIASES.values())
    _write_multi_model_results(
        legacy_project / "results.nc",
        dataset,
        legacy_models,
        [1.0, 2.0, 3.0, 4.0],
    )
    _write_multi_model_results(
        current_project / "results.nc",
        dataset,
        canonical_models,
        [10.0, 20.0, 30.0, 40.0],
    )

    hp_dir = legacy_project / "tables" / "hp_configs"
    hp_dir.mkdir(parents=True)
    for legacy_model in legacy_models:
        hp_path = hp_dir / (
            f"{_safe_path_fragment(dataset)}_{_safe_path_fragment(legacy_model)}.nc"
        )
        xr.Dataset(
            coords={
                "config": [0],
                "task_definition.seq_len": ("config", [10]),
            }
        ).to_netcdf(hp_path)

    loader = ResultsLoader(str(tmp_path))
    legacy_dashboard_dataset = f"{dataset} [legacy_project]"

    for legacy_model, canonical_model in WRAPPER_ALIASES.items():
        assert canonical_model in loader.models
        assert legacy_model not in loader.models
        assert (canonical_model, legacy_model) in loader.applied_model_aliases
        assert (
            loader.xarray_dataset["mean"]
            .sel(
                dataset=legacy_dashboard_dataset,
                model=canonical_model,
                metric_key="test/mae",
            )
            .item()
            == legacy_models.index(legacy_model) + 1
        )
        hp_ds = loader.hp_impact_ds(legacy_dashboard_dataset, canonical_model)
        assert hp_ds is not None
        assert hp_ds.coords["task_definition.seq_len"].item() == 10


class _MemoryResultsLoader(ResultsLoader):
    def __init__(self) -> None:
        dataset = "Dataset A [project]"
        models = ["model_a", "model_b"]
        coords = {"dataset": [dataset], "model": models}
        self.base_dir = "/tmp/memory-results"
        self._expanded_datasets = {}
        self._hp_ds_cache = {}
        self._xarray_dataset = xr.Dataset(
            {
                "opt_mode": xr.DataArray(
                    [["min", "min"]],
                    dims=("dataset", "model"),
                    coords=coords,
                ),
                "sort_metric": xr.DataArray(
                    [["val/loss", "val/loss"]],
                    dims=("dataset", "model"),
                    coords=coords,
                ),
                "opt_metric": xr.DataArray(
                    [["val/loss", "val/loss"]],
                    dims=("dataset", "model"),
                    coords=coords,
                ),
            }
        )
        self._hp_map = {
            (dataset, model): self._make_hp_dataset(offset)
            for model, offset in zip(models, (0.0, 1.0), strict=True)
        }

    @staticmethod
    def _make_hp_dataset(offset: float) -> xr.Dataset:
        return xr.Dataset(
            {
                "mean": xr.DataArray(
                    np.asarray(
                        [
                            [6.0 + offset, 0.6],
                            [4.0 + offset, 0.2],
                            [7.0 + offset, 0.5],
                            [3.0 + offset, 0.1],
                        ]
                    ),
                    dims=("config", "metric"),
                ),
                "std": xr.DataArray(
                    np.full((4, 2), 0.25),
                    dims=("config", "metric"),
                ),
                "count": xr.DataArray(
                    np.full((4, 2), 5.0),
                    dims=("config", "metric"),
                ),
            },
            coords={
                "config": [0, 1, 2, 3],
                "metric": ["test/mae", "val/loss"],
                "task_definition.seq_len": ("config", [10, 10, 50, 50]),
                "optimization.lr": ("config", [0.1, 0.2, 0.1, 0.2]),
            },
        )

    @property
    def xarray_dataset(self) -> xr.Dataset:
        return self._xarray_dataset

    def hp_impact_ds(self, dataset: str, model: str) -> xr.Dataset | None:
        return self._hp_map.get((self.source_dataset(dataset), model))


def test_hp_comparison_builds_deterministic_common_virtual_datasets() -> None:
    loader = _MemoryResultsLoader()

    dimensions, missing_models = loader.hp_comparison_options(
        "Dataset A [project]",
        loader.models,
    )
    labels = loader.configure_hp_comparison(
        "Dataset A [project]",
        ["task_definition.seq_len"],
        loader.models,
    )

    assert dimensions == ["optimization.lr", "task_definition.seq_len"]
    assert missing_models == []
    assert labels == [
        "Dataset A [project] · task_definition.seq_len=10",
        "Dataset A [project] · task_definition.seq_len=50",
    ]
    assert loader.dashboard_datasets == ["Dataset A [project]", *labels]


def test_selected_metric_record_optimizes_within_virtual_dataset_slice() -> None:
    loader = _MemoryResultsLoader()
    labels = loader.configure_hp_comparison(
        "Dataset A [project]",
        ["task_definition.seq_len"],
        loader.models,
    )

    record = loader.selected_metric_record(
        labels[0],
        "model_a",
        metric_key="test/mae",
        sort_metric_key="val/loss",
    )

    assert record["value"] == 4.0
    assert record["std"] == 0.25
    assert record["n"] == 5.0
    assert record["config_index"] == 1
    assert record["matching_configs"] == 2
    assert record["source_dataset"] == "Dataset A [project]"
    assert record["hp_constraints"] == {"task_definition.seq_len": "10"}


def test_hp_comparison_rejects_missing_model_coverage_without_mutating_catalog() -> (
    None
):
    loader = _MemoryResultsLoader()
    initial_catalog = loader.dashboard_datasets

    dimensions, missing_models = loader.hp_comparison_options(
        "Dataset A [project]",
        [*loader.models, "model_without_hp_data"],
    )

    assert dimensions == []
    assert missing_models == ["model_without_hp_data"]
    assert loader.dashboard_datasets == initial_catalog
