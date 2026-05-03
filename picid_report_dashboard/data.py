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
from functools import cached_property
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr


_ALT_METRIC_MODELS = ("tabpfn", "tabdpt", "xgboost")

# ---------------------------------------------------------------------------
# Model name aliases
# Maps old model name → new model name. Aliasing fires ONLY when both sides
# are present in the same loaded dataset; the old name is kept as canonical
# and the new-name slice is merged into it (old data wins on conflict).
# If only one side is present, no renaming occurs.
# ---------------------------------------------------------------------------
MODEL_ALIASES: dict[str, str] = {
    "baselines.crossformer_model.Crossformer_Forecaster": "model.forecasters.crossformer_model.Crossformer_Forecaster",
    "baselines.patchtst_model.PatchTST_Forecaster": "model.forecasters.patchtst_model.PatchTST_Forecaster",
}


def _apply_model_aliases(
    ds: xr.Dataset,
) -> tuple[xr.Dataset, list[tuple[str, str]]]:
    """Apply MODEL_ALIASES only when both old (key) and new (value) names coexist.

    Falls back to the old name as canonical: the new-name slice is merged into
    the old-name slice via combine_first (old data wins on conflict).
    When only one side is present, no renaming occurs.

    Returns (dataset, applied) where applied is a list of (old_name, new_name)
    pairs that were actually unified.
    """
    current_models = set(ds.coords["model"].values)
    # Build new_name → old_name map, but only for pairs where both sides exist.
    active: dict[str, str] = {
        new: old
        for old, new in MODEL_ALIASES.items()
        if old in current_models and new in current_models
    }
    if not active:
        return ds, []

    slices = []
    for model in ds.coords["model"].values:
        if model in active:
            continue  # "new" name — gets folded into the "old" name below
        model_slice = ds.sel(model=model)
        for new_name, old_name in active.items():
            if old_name == model:
                model_slice = model_slice.combine_first(ds.sel(model=new_name))
        slices.append(model_slice.expand_dims("model").assign_coords(model=[model]))

    applied = [(old, new) for new, old in active.items()]
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


class ResultsLoader:
    """Load and merge all results.nc files under a report_output base directory."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self._hp_ds_cache: dict[tuple[str, str], xr.Dataset | None] = {}
        self._applied_model_aliases: list[tuple[str, str]] = []
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
            new_coords = [f"{coord} [{proj_name}]" for coord in ds.coords["dataset"].values]
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
                data.stack(level=0, future_stack=True)
                .dropna(how="all")
                .reset_index()
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
        return pd.concat(frames, ignore_index=True)

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
        return pd.concat(frames, ignore_index=True)

    def hp_impact_ds(
        self, dataset: str, model: str
    ) -> Optional[xr.Dataset]:
        """Load the sorted_aggregated_results xr.Dataset for a (dataset, model) combination.

        ``dataset`` may carry a ' [project_name]' suffix from the dashboard
        coordinate labelling. The suffix is stripped for filename lookup and
        used to restrict the search to the matching project directory.
        """
        cache_key = (dataset, model)
        if cache_key in self._hp_ds_cache:
            return self._hp_ds_cache[cache_key]

        ds_base, project_name = _split_dataset_label(dataset)
        ds_safe = _safe_path_fragment(ds_base)

        # Search only the matching project directory when we know the project
        search_dirs = [
            d for d in self.project_dirs
            if project_name is None or os.path.basename(d) == project_name
        ]

        # Try the given model name first, then any aliased names from MODEL_ALIASES.
        # Aliasing may have renamed the model to an old canonical name while the HP
        # config files on disk were written by the pipeline using the new name.
        model_names_to_try = [model] + [
            new for old, new in MODEL_ALIASES.items() if old == model
        ]
        for model_candidate in model_names_to_try:
            candidate_safe = _safe_path_fragment(model_candidate)
            for d in search_dirs:
                p = os.path.join(d, "tables", "hp_configs", f"{ds_safe}_{candidate_safe}.nc")
                if os.path.isfile(p):
                    loaded = xr.load_dataset(p)
                    self._hp_ds_cache[cache_key] = loaded
                    return loaded

        self._hp_ds_cache[cache_key] = None
        return None

    def metadata_value(self, variable: str, dataset: str, model: str) -> Any:
        """Return a single metadata value from results.nc for one dataset/model."""
        try:
            value = self.xarray_dataset[variable].sel(dataset=dataset, model=model).values
        except Exception:
            return None
        return _normalize_scalar(value)

    def metric_in_hp_dataset(self, hp_ds: xr.Dataset | None, metric_key: str | None) -> bool:
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
        metric_values = np.asarray(hp_ds.sel(metric=metric_key)["mean"].values, dtype=float)
        return bool(np.isfinite(metric_values).any())

    def dataset_has_display_metric(
        self,
        dataset: str,
        *,
        metric_key: str,
        alt_metric_key: str | None = None,
        use_alt_metric: bool = True,
    ) -> bool:
        """Return whether *dataset* has any finite display metric values."""
        for model_name in self.models:
            display_metric_key = self._display_metric_key_for_model(
                model_name,
                metric_key=metric_key,
                alt_metric_key=alt_metric_key,
                use_alt_metric=use_alt_metric,
            )
            hp_ds = self.hp_impact_ds(dataset, model_name)
            if self._metric_has_finite_hp_values(hp_ds, display_metric_key):
                return True
        return False

    def datasets_with_display_metric(
        self,
        *,
        metric_key: str,
        alt_metric_key: str | None = None,
        use_alt_metric: bool = True,
    ) -> list[str]:
        """Return datasets that have at least one finite display metric value."""
        return [
            dataset_name
            for dataset_name in self.datasets
            if self.dataset_has_display_metric(
                dataset_name,
                metric_key=metric_key,
                alt_metric_key=alt_metric_key,
                use_alt_metric=use_alt_metric,
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

        sort_values = np.asarray(hp_ds.sel(metric=sort_metric_key)["mean"].values, dtype=float)
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
        display_metric_key = alt_metric_key if (use_alt and alt_metric_key) else metric_key
        requested_sort_metric_key = (
            alt_sort_metric_key if (use_alt and alt_sort_metric_key) else sort_metric_key
        )

        hp_ds = self.hp_impact_ds(dataset, model)
        opt_metric_key = self.metadata_value("opt_metric", dataset, model)
        stored_sort_metric_key = self.metadata_value("sort_metric", dataset, model)
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
                    available = [str(metric) for metric in hp_ds.coords["metric"].values]
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
            "display_metric_available": self.metric_in_hp_dataset(hp_ds, display_metric_key),
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
        hp_ds = self.hp_impact_ds(dataset, model)
        selection = self.resolve_metric_selection(
            dataset,
            model,
            metric_key=metric_key,
            sort_metric_key=sort_metric_key,
            alt_metric_key=alt_metric_key,
            alt_sort_metric_key=alt_sort_metric_key,
            use_alt_metric=use_alt_metric,
        )
        if _is_empty_hp_dataset(hp_ds) or selection["effective_sort_metric_key"] is None:
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
        record = {
            **selection,
            "value": np.nan,
            "std": np.nan,
            "n": np.nan,
            "config_index": None,
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
    ) -> list[dict[str, Any]]:
        """Return dashboard display records for all dataset/model combinations."""
        datasets = (
            [dataset]
            if dataset is not None
            else self.datasets_with_display_metric(
                metric_key=metric_key,
                alt_metric_key=alt_metric_key,
                use_alt_metric=use_alt_metric,
            )
        )
        records: list[dict[str, Any]] = []
        for dataset_name in datasets:
            for model_name in self.models:
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
