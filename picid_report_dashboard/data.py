"""
Data loading helpers for the piCID dashboard.

Primary source: results.nc (xarray Dataset) written by run_pipeline(output_dir=...).
Secondary: tables/summary.csv, tables/experiment_stats.csv, tables/hp_impact/*.csv.

Usage
-----
loader = ResultsLoader("report_output/")
ds = loader.xarray_dataset          # xr.Dataset, dims (dataset, model, metric_key)
summary = loader.summary_df         # pd.DataFrame
stats = loader.stats_df
hp = loader.hp_impact_ds("n_cmapss", "LSTM")
"""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Collection
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr


_ALT_METRIC_MODELS = ("tabpfn", "tabdpt", "xgboost")

# ---------------------------------------------------------------------------
# Model name aliases
# Maps historical report names to the current package-aligned name. Aliasing
# fires only when both names are present in the loaded dataset; the historical
# slice is merged into the current canonical slice (canonical data wins).
# ---------------------------------------------------------------------------
MODEL_ALIASES: dict[str, str] = {
    "baselines.crossformer_model.Crossformer_Forecaster": "forecasters.crossformer_model.Crossformer_Forecaster",
    "model.forecasters.crossformer_model.Crossformer_Forecaster": "forecasters.crossformer_model.Crossformer_Forecaster",
    "baselines.lstm_model.LSTM_Forecaster": "forecasters.lstm_model.LSTM_Forecaster",
    "baselines.patchtst_model.PatchTST_Forecaster": "forecasters.patchtst_model.PatchTST_Forecaster",
    "model.forecasters.patchtst_model.PatchTST_Forecaster": "forecasters.patchtst_model.PatchTST_Forecaster",
    "baselines.spacetimeformer_model.Spacetimeformer_Forecaster": "forecasters.spacetimeformer_model.Spacetimeformer_Forecaster",
    "baselines.tide_model.TiDE_Forecaster": "forecasters.tide_model.TiDE_Forecaster",
    "baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": "forecasters.timeseries_transformer_model.Timeseries_Transformer_Forecaster",
    "model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": "estimators.cnn1d.wrapper.CNN1D_Wrapper",
    "model.wrappers.mlp_wrapper.MLPWrapper": "estimators.mlp.wrapper.MLPWrapper",
    "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": "estimators.statistical.wrapper.StatisticalBaselineWrapper (exponential)",
    "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": "estimators.statistical.wrapper.StatisticalBaselineWrapper (linear)",
}


def _model_name_candidates(model: str) -> list[str]:
    """Return a model name followed by its canonical and sibling aliases."""
    canonical = MODEL_ALIASES.get(model, model)
    candidates = [
        model,
        canonical,
        *(alias for alias, target in MODEL_ALIASES.items() if target == canonical),
    ]
    return list(dict.fromkeys(candidates))


def _apply_model_aliases(
    ds: xr.Dataset,
) -> tuple[xr.Dataset, list[tuple[str, str]]]:
    """Merge coexisting alias slices into their canonical model slices.

    Returns ``(dataset, applied)`` where each applied entry is a
    ``(canonical_name, alias_name)`` pair.
    """
    current_models = set(ds.coords["model"].values)
    active = {
        alias: canonical
        for alias, canonical in MODEL_ALIASES.items()
        if alias in current_models and canonical in current_models
    }
    if not active:
        return ds, []

    slices = []
    for model in ds.coords["model"].values:
        if model in active:
            continue
        model_slice = ds.sel(model=model)
        for alias, canonical in active.items():
            if canonical == model:
                model_slice = model_slice.combine_first(ds.sel(model=alias))
        slices.append(model_slice.expand_dims("model").assign_coords(model=[model]))

    applied = [(canonical, alias) for alias, canonical in active.items()]
    return xr.concat(slices, dim="model"), applied


def is_alt_model(name: str) -> bool:
    """Return whether *name* should use the alternate metric selectors."""
    lower = name.lower()
    return any(pattern in lower for pattern in _ALT_METRIC_MODELS)


def infer_metric_mode(metric_name: str | None) -> str:
    """Infer whether a metric should be minimized or maximized."""
    if not metric_name:
        return "min"

    metric_lower = metric_name.lower()
    maximize_keywords = (
        "accuracy",
        "acc",
        "f1",
        "f1_score",
        "f_score",
        "auc",
        "roc_auc",
        "precision",
        "recall",
        "r2",
        "r_squared",
        "spearman",
        "pearson",
    )
    minimize_keywords = (
        "loss",
        "error",
        "mse",
        "mean_squared_error",
        "mae",
        "mean_absolute_error",
        "rmse",
        "root_mean_squared_error",
        "mape",
        "mean_absolute_percentage_error",
        "log_loss",
        "cross_entropy",
    )

    if any(keyword in metric_lower for keyword in maximize_keywords):
        return "max"
    if any(keyword in metric_lower for keyword in minimize_keywords):
        return "min"
    return "min"


def _split_dataset_label(dataset: str) -> tuple[str, str | None]:
    """Split 'DatasetName [project_name]' into (base_name, project_name)."""
    if " [" in dataset and dataset.endswith("]"):
        base, project = dataset.rsplit(" [", 1)
        return base, project.rstrip("]")
    return dataset, None


def _safe_path_fragment(value: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(value))[:80]


def _is_empty_hp_dataset(ds: xr.Dataset | None) -> bool:
    """Return whether an HP config dataset has no sortable metric data."""
    if ds is None or not isinstance(ds, xr.Dataset):
        return True
    if "metric" not in ds.coords:
        return True
    if ds.sizes.get("config", 0) == 0:
        return True
    return not bool(ds.data_vars)


def _normalize_scalar(value: Any) -> Any:
    """Convert xarray/numpy scalars to native Python values and map NaN -> None."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


@dataclass(frozen=True)
class ExpandedDataset:
    """One physical dataset constrained to a selected HP-value tuple."""

    source_dataset: str
    dimensions: tuple[str, ...]
    values: tuple[str, ...]
    label: str

    @property
    def constraints(self) -> dict[str, str]:
        return dict(zip(self.dimensions, self.values, strict=True))


class ResultsLoader:
    """Load and merge all results.nc files under a report_output base directory."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self._hp_ds_cache: dict[tuple[str, str], xr.Dataset | None] = {}
        self._applied_model_aliases: list[tuple[str, str]] = []
        self._expanded_datasets: dict[str, ExpandedDataset] = {}
        self._check_base_dir()

    def _check_base_dir(self) -> None:
        if not os.path.isdir(self.base_dir):
            raise FileNotFoundError(
                f"Report output directory not found: {self.base_dir!r}. "
                "Run picid_report with --output-dir first."
            )

    # ------------------------------------------------------------------
    # Project discovery
    # ------------------------------------------------------------------

    @cached_property
    def project_dirs(self) -> list[str]:
        """Subdirectories that contain a results.nc file."""
        paths = sorted(glob.glob(os.path.join(self.base_dir, "*", "results.nc")))
        return [os.path.dirname(p) for p in paths]

    @cached_property
    def project_names(self) -> list[str]:
        """Basename of each project directory that contains a results.nc file."""
        return [os.path.basename(d) for d in self.project_dirs]

    # ------------------------------------------------------------------
    # xarray Dataset (merged across all projects)
    # ------------------------------------------------------------------

    @cached_property
    def xarray_dataset(self) -> xr.Dataset:
        """All results.nc files merged along the 'dataset' dimension.

        When multiple projects share a dataset name (e.g. N-CMAPSS appears in
        both prognostics and diagnostics projects), the project name is appended
        as a suffix to keep coordinates unique.
        """
        if not self.project_dirs:
            raise FileNotFoundError(
                f"No results.nc files found under {self.base_dir!r}."
            )
        relabeled = []
        for proj_dir, proj_name in zip(self.project_dirs, self.project_names):
            ds = xr.open_dataset(os.path.join(proj_dir, "results.nc"))
            new_coords = [
                f"{coord} [{proj_name}]" for coord in ds.coords["dataset"].values
            ]
            ds = ds.assign_coords(dataset=new_coords)
            relabeled.append(ds)

        combined = xr.concat(
            relabeled, dim="dataset", join="outer", combine_attrs="drop_conflicts"
        )
        # Force eager loading before closing file handles.  xr.open_dataset is
        # lazy by default; closing the source files while combined still holds
        # deferred references causes Bad-file-descriptor errors on first access.
        combined.load()
        for ds in relabeled:
            ds.close()
        combined, self._applied_model_aliases = _apply_model_aliases(combined)
        return combined

    # ------------------------------------------------------------------
    # Convenience accessors derived from the xarray Dataset
    # ------------------------------------------------------------------

    @property
    def datasets(self) -> list[str]:
        """Unique dataset coordinate labels (may include project suffix for duplicates)."""
        return list(self.xarray_dataset.coords["dataset"].values)

    @property
    def dashboard_datasets(self) -> list[str]:
        """Physical datasets followed by configured HP-constrained datasets."""
        return [str(dataset) for dataset in self.datasets] + list(
            getattr(self, "_expanded_datasets", {})
        )

    def expanded_dataset(self, dataset: str) -> ExpandedDataset | None:
        """Return the expansion specification for a virtual dataset label."""
        return getattr(self, "_expanded_datasets", {}).get(str(dataset))

    def source_dataset(self, dataset: str) -> str:
        """Resolve a dashboard dataset label to its physical source dataset."""
        expanded = self.expanded_dataset(dataset)
        return expanded.source_dataset if expanded is not None else str(dataset)

    def clear_hp_comparison(self) -> None:
        """Remove all configured virtual HP datasets."""
        self._expanded_datasets = {}

    def hp_comparison_options(
        self,
        source_dataset: str,
        models: Collection[str],
    ) -> tuple[list[str], list[str]]:
        """Return common varying HP dimensions and models without HP data."""
        coord_sets: list[set[str]] = []
        datasets: list[xr.Dataset] = []
        missing_models: list[str] = []
        for model in models:
            hp_ds = self.hp_impact_ds(source_dataset, str(model))
            if _is_empty_hp_dataset(hp_ds):
                missing_models.append(str(model))
                continue
            assert hp_ds is not None
            coord_sets.append(
                {
                    str(coord)
                    for coord in hp_ds.coords
                    if coord not in {"config", "metric", "Model"}
                    and hp_ds.coords[coord].dims == ("config",)
                }
            )
            datasets.append(hp_ds)
        if missing_models or not coord_sets:
            return [], missing_models

        common_dimensions = set.intersection(*coord_sets)
        varying_dimensions = []
        for dimension in sorted(common_dimensions):
            values = {
                str(_normalize_scalar(value))
                for hp_ds in datasets
                for value in hp_ds.coords[dimension].values.tolist()
            }
            if len(values) > 1:
                varying_dimensions.append(dimension)
        return varying_dimensions, []

    def configure_hp_comparison(
        self,
        source_dataset: str,
        dimensions: Collection[str],
        models: Collection[str],
    ) -> list[str]:
        """Create comparable virtual datasets for common selected HP tuples."""
        normalized_source = str(source_dataset)
        normalized_dimensions = tuple(dict.fromkeys(str(name) for name in dimensions))
        normalized_models = [str(model) for model in models]
        self.clear_hp_comparison()
        if not normalized_dimensions:
            return []
        if normalized_source not in self.datasets:
            raise ValueError(f"Dataset {normalized_source!r} is not available.")

        tuple_sets: list[set[tuple[str, ...]]] = []
        missing_models: list[str] = []
        for model in normalized_models:
            hp_ds = self.hp_impact_ds(normalized_source, model)
            if _is_empty_hp_dataset(hp_ds) or any(
                dimension not in hp_ds.coords for dimension in normalized_dimensions
            ):
                missing_models.append(model)
                continue
            assert hp_ds is not None
            tuple_sets.append(
                {
                    tuple(
                        str(_normalize_scalar(hp_ds.coords[dimension].values[index]))
                        for dimension in normalized_dimensions
                    )
                    for index in range(hp_ds.sizes["config"])
                }
            )
        if missing_models:
            missing = ", ".join(missing_models)
            raise ValueError(
                "HP comparison is unavailable for these enabled models: "
                f"{missing}. Disable them or choose another dataset."
            )
        if not tuple_sets:
            raise ValueError("No HP configurations are available for this dataset.")

        common_tuples = sorted(set.intersection(*tuple_sets))
        if not common_tuples:
            raise ValueError(
                "The enabled models do not share any selected HP-value tuples."
            )
        for values in common_tuples:
            suffix = " · ".join(
                f"{dimension}={value}"
                for dimension, value in zip(
                    normalized_dimensions,
                    values,
                    strict=True,
                )
            )
            label = f"{normalized_source} · {suffix}"
            self._expanded_datasets[label] = ExpandedDataset(
                source_dataset=normalized_source,
                dimensions=normalized_dimensions,
                values=values,
                label=label,
            )
        return list(self._expanded_datasets)

    def preview_hp_comparison(
        self,
        source_dataset: str,
        dimensions: Collection[str],
        models: Collection[str],
    ) -> list[str]:
        """Return prospective virtual labels without changing the active catalog."""
        active = self._expanded_datasets
        try:
            return self.configure_hp_comparison(
                source_dataset,
                dimensions,
                models,
            )
        finally:
            self._expanded_datasets = active

    def filter_hp_configs(self, dataset: str, model: str) -> xr.Dataset | None:
        """Return HP configs constrained by a virtual dataset specification."""
        hp_ds = self.hp_impact_ds(self.source_dataset(dataset), model)
        expanded = self.expanded_dataset(dataset)
        if _is_empty_hp_dataset(hp_ds) or expanded is None:
            return hp_ds
        assert hp_ds is not None
        mask = np.ones(hp_ds.sizes["config"], dtype=bool)
        for dimension, value in expanded.constraints.items():
            if dimension not in hp_ds.coords:
                return hp_ds.isel(config=[])
            coord_values = np.asarray(
                [
                    str(_normalize_scalar(coord_value))
                    for coord_value in hp_ds.coords[dimension].values.tolist()
                ]
            )
            mask &= coord_values == value
        return hp_ds.isel(config=np.flatnonzero(mask).tolist())

    @property
    def models(self) -> list[str]:
        """Model names present across all loaded projects."""
        return list(self.xarray_dataset.coords["model"].values)

    @property
    def applied_model_aliases(self) -> list[tuple[str, str]]:
        """(old_name, new_name) pairs that were actually unified by MODEL_ALIASES."""
        return list(self._applied_model_aliases)

    @property
    def metric_keys(self) -> list[str]:
        """All metric keys in 'prefix/metric_name' form."""
        return list(self.xarray_dataset.coords["metric_key"].values)

    def metric_matrix(
        self,
        metric_key: str,
        stat: str = "mean",
    ) -> pd.DataFrame:
        """Return a (dataset × model) DataFrame for *metric_key*.

        Parameters
        ----------
        metric_key:
            Key in the form ``"prefix/metric_name"``, e.g. ``"test/mae_denormalized"``.
        stat:
            One of ``"mean"``, ``"std"``, ``"n"``.
        """
        ds = self.xarray_dataset
        arr = ds[stat].sel(metric_key=metric_key).values  # shape (n_datasets, n_models)
        return pd.DataFrame(arr, index=self.datasets, columns=self.models)

    def mean_std_df(self, metric_key: str) -> pd.DataFrame:
        """Return a (dataset × model) DataFrame with 'mean ± std' strings."""
        mean = self.metric_matrix(metric_key, "mean")
        std = self.metric_matrix(metric_key, "std")
        result = mean.copy().astype(object)
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                m, s = mean.iloc[i, j], std.iloc[i, j]
                if np.isnan(m):
                    result.iloc[i, j] = "—"
                elif np.isnan(s):
                    result.iloc[i, j] = f"{m:.4f}"
                else:
                    result.iloc[i, j] = f"{m:.4f} ± {s:.4f}"
        return result

    def metadata_df(self, variable: str) -> pd.DataFrame:
        """Return a (dataset × model) DataFrame for a metadata variable.

        Available variables: sort_metric, opt_metric, opt_mode, opt_value,
        total_runs, configs_failed_seed, configs_failed_metric.
        """
        ds = self.xarray_dataset
        arr = ds[variable].values
        return pd.DataFrame(arr, index=self.datasets, columns=self.models)

    # ------------------------------------------------------------------
    # CSV table accessors
    # ------------------------------------------------------------------

    def _canonicalize_model_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Match CSV model names to the canonical names in ``xarray_dataset``."""
        _ = self.xarray_dataset
        alias_to_canonical = {
            alias: canonical for canonical, alias in self.applied_model_aliases
        }
        if alias_to_canonical and "Model" in df.columns:
            df["Model"] = df["Model"].replace(alias_to_canonical)
        return df

    @cached_property
    def summary_df(self) -> Optional[pd.DataFrame]:
        """Merged summary.csv reshaped to long form (Dataset, Metric, Model, Value).

        Each summary.csv has a 2-row header (row 0 = dataset names, row 1 = metric
        names) followed by a spurious 'Model' label row, then one row per model.
        Naively concatting 13 files with all-different column names produces a
        189×393 DataFrame that is 90% NaN. Instead we parse with header=[0,1] and
        melt to a clean 4-column long-form table.
        """
        frames = []
        for d in self.project_dirs:
            p = os.path.join(d, "tables", "summary.csv")
            if not os.path.isfile(p):
                continue
            df = pd.read_csv(p, header=[0, 1])
            first_col = df.columns[0]  # e.g. ("Dataset", "Metric")
            # Drop the spurious "Model" label row
            df = df[df[first_col] != "Model"].reset_index(drop=True)
            # Use model names as row index, drop the meta column
            data = df.iloc[:, 1:].copy()
            data.index = df[first_col].values
            data.index.name = "Model"
            # Stack the dataset level (level 0) into rows → index becomes (Model, Dataset).
            # future_stack=True opts into the new pandas-2 implementation; dropna(how="all")
            # preserves the old default of dropping (Model, Dataset) pairs with no metrics.
            stacked = (
                data.stack(level=0, future_stack=True).dropna(how="all").reset_index()
            )
            stacked.columns = ["Model", "Dataset"] + list(stacked.columns[2:])
            # Melt the remaining metric columns into long form
            metric_cols = [c for c in stacked.columns if c not in ("Model", "Dataset")]
            melted = stacked.melt(
                id_vars=["Model", "Dataset"],
                value_vars=metric_cols,
                var_name="Metric",
                value_name="Value",
            )
            melted["Project"] = os.path.basename(d)
            frames.append(melted[["Project", "Dataset", "Metric", "Model", "Value"]])
        if not frames:
            return None
        return self._canonicalize_model_column(pd.concat(frames, ignore_index=True))

    @cached_property
    def stats_df(self) -> Optional[pd.DataFrame]:
        """Merged experiment_stats.csv."""
        frames = []
        for d in self.project_dirs:
            p = os.path.join(d, "tables", "experiment_stats.csv")
            if os.path.isfile(p):
                df = pd.read_csv(p)
                df.insert(0, "Project", os.path.basename(d))
                frames.append(df)
        if not frames:
            return None
        return self._canonicalize_model_column(pd.concat(frames, ignore_index=True))

    def hp_impact_ds(self, dataset: str, model: str) -> Optional[xr.Dataset]:
        """Load the sorted_aggregated_results xr.Dataset for a (dataset, model) combination.

        ``dataset`` may carry a ' [project_name]' suffix from the dashboard
        coordinate labelling. The suffix is stripped for filename lookup and
        used to restrict the search to the matching project directory.
        """
        dataset = self.source_dataset(dataset)
        cache_key = (dataset, model)
        if cache_key in self._hp_ds_cache:
            return self._hp_ds_cache[cache_key]

        ds_base, project_name = _split_dataset_label(dataset)
        ds_safe = _safe_path_fragment(ds_base)

        # Search only the matching project directory when we know the project
        search_dirs = [
            d
            for d in self.project_dirs
            if project_name is None or os.path.basename(d) == project_name
        ]

        # The merged results use canonical model names, while per-project HP
        # files retain whichever module path was active when the report ran.
        model_names_to_try = _model_name_candidates(model)
        for model_candidate in model_names_to_try:
            candidate_safe = _safe_path_fragment(model_candidate)
            for d in search_dirs:
                p = os.path.join(
                    d, "tables", "hp_configs", f"{ds_safe}_{candidate_safe}.nc"
                )
                if os.path.isfile(p):
                    loaded = xr.load_dataset(p)
                    self._hp_ds_cache[cache_key] = loaded
                    return loaded

        self._hp_ds_cache[cache_key] = None
        return None

    def metadata_value(self, variable: str, dataset: str, model: str) -> Any:
        """Return a single metadata value from results.nc for one dataset/model."""
        try:
            value = (
                self.xarray_dataset[variable]
                .sel(
                    dataset=self.source_dataset(dataset),
                    model=model,
                )
                .values
            )
        except Exception:
            return None
        return _normalize_scalar(value)

    def metric_in_hp_dataset(
        self, hp_ds: xr.Dataset | None, metric_key: str | None
    ) -> bool:
        """Return whether *metric_key* exists in the given HP config dataset."""
        if metric_key is None or _is_empty_hp_dataset(hp_ds):
            return False
        return metric_key in {str(metric) for metric in hp_ds.coords["metric"].values}

    def _display_metric_key_for_model(
        self,
        model: str,
        *,
        metric_key: str,
        alt_metric_key: str | None,
        use_alt_metric: bool,
    ) -> str:
        """Resolve the display metric key for one model under the active metric mode."""
        if use_alt_metric and alt_metric_key and is_alt_model(model):
            return alt_metric_key
        return metric_key

    def _metric_has_finite_hp_values(
        self, hp_ds: xr.Dataset | None, metric_key: str | None
    ) -> bool:
        """Return whether the HP dataset contains any finite values for *metric_key*."""
        if not self.metric_in_hp_dataset(hp_ds, metric_key):
            return False
        metric_values = np.asarray(
            hp_ds.sel(metric=metric_key)["mean"].values, dtype=float
        )
        return bool(np.isfinite(metric_values).any())

    def dataset_has_display_metric(
        self,
        dataset: str,
        *,
        metric_key: str,
        alt_metric_key: str | None = None,
        use_alt_metric: bool = True,
        enabled_models: Collection[str] | None = None,
    ) -> bool:
        """Return whether *dataset* has any finite display metric values."""
        models = self.models
        if enabled_models is not None:
            enabled = set(enabled_models)
            models = [model_name for model_name in models if model_name in enabled]

        for model_name in models:
            display_metric_key = self._display_metric_key_for_model(
                model_name,
                metric_key=metric_key,
                alt_metric_key=alt_metric_key,
                use_alt_metric=use_alt_metric,
            )
            hp_ds = self.filter_hp_configs(dataset, model_name)
            if self._metric_has_finite_hp_values(hp_ds, display_metric_key):
                return True
        return False

    def datasets_with_display_metric(
        self,
        *,
        metric_key: str,
        alt_metric_key: str | None = None,
        use_alt_metric: bool = True,
        enabled_models: Collection[str] | None = None,
    ) -> list[str]:
        """Return datasets that have at least one finite display metric value."""
        return [
            dataset_name
            for dataset_name in self.dashboard_datasets
            if self.dataset_has_display_metric(
                dataset_name,
                metric_key=metric_key,
                alt_metric_key=alt_metric_key,
                use_alt_metric=use_alt_metric,
                enabled_models=enabled_models,
            )
        ]

    def sort_config_indices(
        self,
        hp_ds: xr.Dataset | None,
        sort_metric_key: str | None,
        sort_mode: str,
    ) -> np.ndarray:
        """Return config indices sorted by *sort_metric_key* with NaNs placed last."""
        if (
            _is_empty_hp_dataset(hp_ds)
            or sort_metric_key is None
            or not self.metric_in_hp_dataset(hp_ds, sort_metric_key)
        ):
            size = 0 if hp_ds is None else hp_ds.sizes.get("config", 0)
            return np.arange(size, dtype=int)

        sort_values = np.asarray(
            hp_ds.sel(metric=sort_metric_key)["mean"].values, dtype=float
        )
        valid_mask = np.isfinite(sort_values)
        valid_indices = np.flatnonzero(valid_mask)
        invalid_indices = np.flatnonzero(~valid_mask)

        ordered_valid = valid_indices[np.argsort(sort_values[valid_indices])]
        if sort_mode != "min":
            ordered_valid = ordered_valid[::-1]
        return np.concatenate([ordered_valid, invalid_indices]).astype(int, copy=False)

    def resolve_metric_selection(
        self,
        dataset: str,
        model: str,
        *,
        metric_key: str,
        sort_metric_key: str | None,
        alt_metric_key: str | None = None,
        alt_sort_metric_key: str | None = None,
        use_alt_metric: bool = True,
    ) -> dict[str, Any]:
        """Resolve effective display/sort metrics for one dataset/model pair."""
        use_alt = bool(use_alt_metric and is_alt_model(model))
        display_metric_key = (
            alt_metric_key if (use_alt and alt_metric_key) else metric_key
        )
        requested_sort_metric_key = (
            alt_sort_metric_key
            if (use_alt and alt_sort_metric_key)
            else sort_metric_key
        )

        hp_ds = self.filter_hp_configs(dataset, model)
        source_dataset = self.source_dataset(dataset)
        opt_metric_key = self.metadata_value("opt_metric", source_dataset, model)
        stored_sort_metric_key = self.metadata_value(
            "sort_metric", source_dataset, model
        )
        effective_sort_metric_key = (
            requested_sort_metric_key or opt_metric_key or stored_sort_metric_key
        )
        fallback_from_sort_metric_key = None

        if not self.metric_in_hp_dataset(hp_ds, effective_sort_metric_key):
            # Prefer the report's stored sort metric when the requested selector
            # value is unavailable. Falling back to the optimization metric is a
            # last resort because it may differ from the metric used to rank the
            # report's best configuration.
            for candidate in (stored_sort_metric_key, opt_metric_key):
                if self.metric_in_hp_dataset(hp_ds, candidate):
                    fallback_from_sort_metric_key = effective_sort_metric_key
                    effective_sort_metric_key = candidate
                    break
            else:
                if not _is_empty_hp_dataset(hp_ds):
                    available = [
                        str(metric) for metric in hp_ds.coords["metric"].values
                    ]
                    if available:
                        fallback_from_sort_metric_key = effective_sort_metric_key
                        effective_sort_metric_key = available[0]
                    else:
                        effective_sort_metric_key = None
                else:
                    effective_sort_metric_key = effective_sort_metric_key

        return {
            "dataset": dataset,
            "model": model,
            "uses_alt_metric": use_alt,
            "display_metric_key": display_metric_key,
            "requested_sort_metric_key": requested_sort_metric_key,
            "effective_sort_metric_key": effective_sort_metric_key,
            "sort_mode": infer_metric_mode(effective_sort_metric_key),
            "sort_metric_fell_back": (
                fallback_from_sort_metric_key is not None
                and fallback_from_sort_metric_key != effective_sort_metric_key
            ),
            "fallback_from_sort_metric_key": fallback_from_sort_metric_key,
            "opt_metric_key": opt_metric_key,
            "stored_sort_metric_key": stored_sort_metric_key,
            "hp_dataset_available": not _is_empty_hp_dataset(hp_ds),
            "display_metric_available": self.metric_in_hp_dataset(
                hp_ds, display_metric_key
            ),
        }

    def sorted_hp_configs(
        self,
        dataset: str,
        model: str,
        *,
        metric_key: str,
        sort_metric_key: str | None,
        alt_metric_key: str | None = None,
        alt_sort_metric_key: str | None = None,
        use_alt_metric: bool = True,
    ) -> tuple[xr.Dataset | None, dict[str, Any]]:
        """Return the HP dataset sorted by the active dashboard sort metric."""
        hp_ds = self.filter_hp_configs(dataset, model)
        selection = self.resolve_metric_selection(
            dataset,
            model,
            metric_key=metric_key,
            sort_metric_key=sort_metric_key,
            alt_metric_key=alt_metric_key,
            alt_sort_metric_key=alt_sort_metric_key,
            use_alt_metric=use_alt_metric,
        )
        if (
            _is_empty_hp_dataset(hp_ds)
            or selection["effective_sort_metric_key"] is None
        ):
            return hp_ds, selection

        sort_idx = self.sort_config_indices(
            hp_ds,
            selection["effective_sort_metric_key"],
            selection["sort_mode"],
        )
        return hp_ds.isel(config=sort_idx.tolist()), selection

    def selected_metric_record(
        self,
        dataset: str,
        model: str,
        *,
        metric_key: str,
        sort_metric_key: str | None,
        alt_metric_key: str | None = None,
        alt_sort_metric_key: str | None = None,
        use_alt_metric: bool = True,
    ) -> dict[str, Any]:
        """Return display metric stats from the best config under the active sort metric."""
        sorted_hp_ds, selection = self.sorted_hp_configs(
            dataset,
            model,
            metric_key=metric_key,
            sort_metric_key=sort_metric_key,
            alt_metric_key=alt_metric_key,
            alt_sort_metric_key=alt_sort_metric_key,
            use_alt_metric=use_alt_metric,
        )
        expanded = self.expanded_dataset(dataset)
        record = {
            **selection,
            "value": np.nan,
            "std": np.nan,
            "n": np.nan,
            "config_index": None,
            "source_dataset": self.source_dataset(dataset),
            "hp_constraints": expanded.constraints if expanded is not None else {},
            "matching_configs": (
                0 if sorted_hp_ds is None else sorted_hp_ds.sizes.get("config", 0)
            ),
        }
        if _is_empty_hp_dataset(sorted_hp_ds):
            return record
        if sorted_hp_ds.sizes.get("config", 0) == 0:
            return record

        best_cfg = sorted_hp_ds.isel(config=0)
        record["config_index"] = int(best_cfg.coords["config"].item())
        display_metric_key = selection["display_metric_key"]
        if not self.metric_in_hp_dataset(sorted_hp_ds, display_metric_key):
            return record

        metric_slice = best_cfg.sel(metric=display_metric_key)
        mean_value = float(metric_slice["mean"].values)
        std_value = float(metric_slice["std"].values)
        count_value = float(metric_slice["count"].values)

        record["value"] = mean_value
        record["std"] = np.nan if np.isnan(std_value) else std_value
        record["n"] = np.nan if np.isnan(count_value) else count_value
        return record

    def selected_metric_records(
        self,
        *,
        metric_key: str,
        sort_metric_key: str | None,
        alt_metric_key: str | None = None,
        alt_sort_metric_key: str | None = None,
        use_alt_metric: bool = True,
        dataset: str | None = None,
        enabled_datasets: Collection[str] | None = None,
        enabled_models: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return dashboard display records for the requested datasets and models."""
        if dataset is not None:
            datasets = [dataset]
        else:
            datasets = self.datasets_with_display_metric(
                metric_key=metric_key,
                alt_metric_key=alt_metric_key,
                use_alt_metric=use_alt_metric,
                enabled_models=enabled_models,
            )
            if enabled_datasets is not None:
                enabled = set(enabled_datasets)
                datasets = [name for name in datasets if name in enabled]

        models = self.models
        if enabled_models is not None:
            enabled = set(enabled_models)
            models = [model_name for model_name in models if model_name in enabled]
        records: list[dict[str, Any]] = []
        for dataset_name in datasets:
            for model_name in models:
                records.append(
                    self.selected_metric_record(
                        dataset_name,
                        model_name,
                        metric_key=metric_key,
                        sort_metric_key=sort_metric_key,
                        alt_metric_key=alt_metric_key,
                        alt_sort_metric_key=alt_sort_metric_key,
                        use_alt_metric=use_alt_metric,
                    )
                )
        return records
