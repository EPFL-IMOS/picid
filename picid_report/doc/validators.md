# validators.py — Schema and seed validation

**Navigation:** [← Documentation index](README.md)

**Module:** `picid_report.core.validators`

This document describes every **validation** function: schema check, seed filtering, modification logging, and hidden-variation checks. Each function is documented with **examples**.

---

## 1. validate_schema(df, required_cols)

Checks that all **required_cols** are present in the DataFrame. If any are missing, logs an error and **raises ValueError**.

**Example:**
```python
from picid_report import config
from picid_report.validators import validate_schema

validate_schema(df, config.REQUIRED_COLUMNS)
# Passes silently if all required columns exist
# Raises ValueError with a clear message listing missing columns if not
```

**Example — custom required list:**
```python
validate_schema(df, ["model._target_", "datasource.data_name", "seed"])
```

---

## 2. log_modification(action, reason, input_shape, output_shape, context=None, level=logging.INFO)

Logs a DataFrame shape change (e.g. after subsetting or seed filter). Only logs if rows or columns actually changed. Used internally by analysis and validators.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `action` | Short description (e.g. "Seed Filter", "Subsetting DataFrame"). |
| `reason` | Justification for the change. |
| `input_shape` | (rows, cols) before. |
| `output_shape` | (rows, cols) after. |
| `context` | Optional (e.g. "Model='LSTM' \| Dataset='UNIBO21'"). |
| `level` | Log level; use `logging.DEBUG` for high-frequency logs. |

**Example:**
```python
import logging
from picid_report.validators import log_modification

log_modification(
    action="Strict Seed Filter applied on 'seed'",
    reason="Dropping groups that do not contain all required seeds: {72, 88, 101}",
    input_shape=(1000, 50),
    output_shape=(800, 50),
    context="dataset=UNIBO21, model=LSTM",
)
# Logs: [DATAFRAME MODIFICATION] ..., Context: ..., Reason: ..., Shape: (1000, 50) -> (800, 50), Change: 200 rows dropped
```

**Example — DEBUG level (e.g. per-dataset/model subsetting):**
```python
log_modification(
    action="Subsetting DataFrame",
    reason="Filtering to current dataset/model",
    input_shape=df.shape,
    output_shape=subset.shape,
    context=f"Model='{model}' | Dataset='{dataset}'",
    level=logging.DEBUG,
)
```

---

## 3. validate_seeds(df, group_cols, seed_col, required_seeds=None, allow_fallback=False, context=None)

Filters the DataFrame to **groups** (defined by **group_cols**) that contain **all** values in **required_seeds** for **seed_col**. Groups missing any required seed are dropped. If **required_seeds** is None or **seed_col** is missing, returns `df` unchanged.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `df` | DataFrame with seed_col and group_cols. |
| `group_cols` | Columns to group by (e.g. `["Model", "task_definition.seq_len", "optimization.lr"]`). |
| `seed_col` | Column name for the seed (e.g. `"seed"`, `"datasource.parameters.data_seed"`). |
| `required_seeds` | Set of seed values that must all be present in each group; **None** = no filtering. |
| `allow_fallback` | If True and the filter would remove all rows, returns original `df` and logs a warning. |
| `context` | Optional string for warning messages (e.g. `"dataset=UNIBO21, model=LSTM"`). |

**Returns:** Filtered DataFrame (or original if no required_seeds, seed_col missing, or fallback triggered).

**Example — filter to groups with all model seeds:**
```python
from picid_report.validators import validate_seeds

subset = validate_seeds(
    subset,
    group_cols=["Model", "task_definition.seq_len", "optimization.lr"],
    seed_col="seed",
    required_seeds={72, 88, 101, 666, 226688},
    allow_fallback=True,
    context="dataset=UNIBO21, model=LSTM",
)
# Only configs that have runs for all 5 seeds remain
```

**Example — no filtering when required_seeds is None:**
```python
subset = validate_seeds(subset, group_cols=["Model"], seed_col="seed", required_seeds=None)
# Returns df unchanged
```

**Example — seed column missing:**
```python
subset = validate_seeds(subset, group_cols=["Model"], seed_col="seed", required_seeds={1, 2, 3})
# Logs warning and returns df unchanged
```

---

## 4. check_hidden_variations(df, group_cols, base_ignored_cols, additional_ignored_cols=None)

Warns when a **non-HP** column varies within a group (e.g. same dataset/model/HP config but different value in another column). Columns in **group_cols** or matching **base_ignored_cols** (substrings) or **additional_ignored_cols** are ignored; numeric columns are skipped. Logs a warning per column that varies within any group.

**Example:**
```python
from picid_report import config
from picid_report.validators import check_hidden_variations

check_hidden_variations(
    subset,
    group_cols=["Model", "task_definition.seq_len", "optimization.lr"],
    base_ignored_cols=config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH + ["seed", "datasource.parameters.data_seed"],
    additional_ignored_cols=["gpu_id", "slurm_job_id"],
)
# No return value; logs warnings for any non-ignored, non-numeric column that has >1 unique value per group
```

---

## 5. Where they are used in the pipeline

- **validate_schema:** Called in **run.py** after preprocess with `config.REQUIRED_COLUMNS`.
- **validate_seeds:** Called in **analysis.py** twice per (dataset, model): first for data seed (allow_fallback=False), then for model seed (allow_fallback=True); uses **log_modification** to log row drops.
- **log_modification:** Used by analysis (subsetting) and validators (seed filter).
- **check_hidden_variations:** Called in **analysis.py** per (dataset, model) after seed validation to warn about hidden variation.

See [analysis.md](analysis.md) and [run.md](run.md) for the full flow.

---

**Navigation:** [← Documentation index](README.md)
