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
hp = loader.hp_impact_df("n_cmapss", "LSTM")
"""

from __future__ import annotations

import glob
import os
import re
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


class ResultsLoader:
    """Load and merge all results.nc files under a report_output base directory."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = os.path.abspath(base_dir)
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
        seen: dict[str, int] = {}  # dataset_name -> count of appearances
        relabeled = []
        for proj_dir, proj_name in zip(self.project_dirs, self.project_names):
            ds = xr.open_dataset(os.path.join(proj_dir, "results.nc"))
            # Rename duplicate dataset coordinates to include the project name
            new_coords = []
            for coord in ds.coords["dataset"].values:
                key = str(coord)
                if key in seen:
                    new_coords.append(f"{key} [{proj_name}]")
                else:
                    seen[key] = 1
                    new_coords.append(key)
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
            # Stack the dataset level (level 0) into rows → index becomes (Model, Dataset)
            stacked = data.stack(level=0).reset_index()
            stacked.columns = ["Model", "Dataset"] + list(stacked.columns[2:])
            # Melt the remaining metric columns into long form
            metric_cols = [c for c in stacked.columns if c not in ("Model", "Dataset")]
            melted = stacked.melt(
                id_vars=["Model", "Dataset"],
                value_vars=metric_cols,
                var_name="Metric",
                value_name="Value",
            )
            frames.append(melted[["Dataset", "Metric", "Model", "Value"]])
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
                frames.append(pd.read_csv(p))
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def hp_impact_df(
        self, dataset: str, model: str
    ) -> Optional[pd.DataFrame]:
        """Load the HP impact CSV for a (dataset, model) combination.

        Searches all project HP impact directories for a matching filename.
        """
        def _safe(s: str) -> str:
            return re.sub(r"[^\w\-.]", "_", str(s))[:80]

        filename = f"{_safe(dataset)}_{_safe(model)}.csv"
        for d in self.project_dirs:
            hp_dir = os.path.join(d, "tables", "hp_impact")
            p = os.path.join(hp_dir, filename)
            if os.path.isfile(p):
                return pd.read_csv(p)
        return None
