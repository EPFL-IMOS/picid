# run_processor.py — Load runs from W&B or cache

**Navigation:** [← Documentation index](README.md)

**Module:** `picid_report.core.run_processor`

This document describes the data-loading step: **`load_runs_df()`**. It is responsible for fetching run data from Weights & Biases (or a local CSV cache) and turning it into a flat, cleaned pandas DataFrame plus metadata for the analysis stage.

---

## 1. Overview

The module performs:

1. **Fetch or load** — If a CSV cache exists for the project, load from it; otherwise fetch runs from the W&B API (finished runs only), merge `summary` and `config`, and save to CSV.
2. **Normalize** — Flatten nested dict columns (e.g. `model` → `model._target_`, `model.lr`) using the config’s list of columns to normalize.
3. **Clean** — Drop columns that are >90% NaN; drop columns whose names match `COLUMN_FILTERS_TO_DROP` (e.g. `paths.*`).
4. **Return** — DataFrame, list of config column names (from normalization), and list of dropped column names.

If **`pipeline_config`** is provided, its `columns_to_normalize` and `column_filters_to_drop` are used; otherwise module-level `config.COLUMNS_TO_NORMALIZE` and `config.COLUMN_FILTERS_TO_DROP` are used.

---

## 2. load_runs_df()

**Signature:**

```python
def load_runs_df(
    project_name: str,
    user: str,
    csv_cache_dir: str = "csv_files",
    pipeline_config: Optional[PipelineConfig] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `project_name` | W&B project name (e.g. `'my-awesome-project'`). |
| `user` | W&B username or entity (e.g. `'my-team'`). |
| `csv_cache_dir` | Directory to store and look for cached CSV files. |
| `pipeline_config` | Optional. If set, uses its `columns_to_normalize` and `column_filters_to_drop`; otherwise uses module `config`. |

**Returns:**

- **DataFrame** — One row per run; merged summary + config, flattened; high-NaN and filter-matched columns dropped.
- **List[str]** — Config column names (those created during normalization), used later to distinguish config vs metric columns.
- **List[str]** — Column names that were dropped (NaN + filter-based).

---

### Example 1: Load with defaults (module config, cache in `csv_files`)

```python
from picid_report import load_runs_df

df, config_columns, dropped_columns = load_runs_df(
    project_name="29_01_2026_unibo_prognostics_combined",
    user="anonlab-buildingenergy-1",
    csv_cache_dir="csv_files",
)
# If csv_files/29_01_2026_unibo_prognostics_combined.csv exists: load from cache
# Else: fetch from W&B, save to that path
print(df.shape)
print("Config columns count:", len(config_columns))
print("Dropped count:", len(dropped_columns))
```

---

### Example 2: Force fresh fetch (delete cache first)

```python
import os
from picid_report import load_runs_df

cache_path = "csv_files/my_project.csv"
if os.path.exists(cache_path):
    os.remove(cache_path)

df, config_columns, dropped_columns = load_runs_df(
    project_name="my_project",
    user="my_team",
    csv_cache_dir="csv_files",
)
# Always fetches from W&B and overwrites cache
```

---

### Example 3: Use PipelineConfig to override normalize/drop lists

```python
from picid_report.config import PipelineConfig
from picid_report import config
from picid_report import load_runs_df

cfg = PipelineConfig.from_default()
# Drop more columns (e.g. add "logger.")
cfg.column_filters_to_drop = list(config.COLUMN_FILTERS_TO_DROP) + ["logger."]

df, config_columns, dropped_columns = load_runs_df(
    project_name="my_project",
    user="my_team",
    csv_cache_dir="csv_files",
    pipeline_config=cfg,
)
```

---

### Example 4: Different cache directory per experiment

```python
df, config_columns, dropped_columns = load_runs_df(
    project_name="exp_2026_02",
    user="my_team",
    csv_cache_dir="cache/exp_2026_02",
)
# Cache file: cache/exp_2026_02/exp_2026_02.csv
```

---

## 3. Internal helpers

- **`_filter_columns(df, filters)`** — Returns column names that contain any of the filter substrings. Used to find columns to drop.
- **`_normalize_column(df, column_to_normalize)`** — Flattens one nested-dict column into `parent.key` columns via `pd.json_normalize`; handles stringified dicts with `ast.literal_eval`. Returns updated DataFrame and list of new column names.

These are used by `load_runs_df()`; you typically do not call them directly.

---

**Navigation:** [← Documentation index](README.md)
