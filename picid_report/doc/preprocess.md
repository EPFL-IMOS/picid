# preprocess.py — Model name cleaning

**Navigation:** [← Documentation index](README.md)

**Module:** `picid_report.core.preprocess`

This document describes the **preprocessing** step used by the pipeline: **`clean_and_rename_models()`**. It normalizes model names for display and grouping.

---

## 1. clean_and_rename_models()

**Signature:**

```python
def clean_and_rename_models(
    df: pd.DataFrame,
    model_target_col: Optional[str] = None,
    model_type_col: Optional[str] = None,
) -> pd.DataFrame:
```

**Behavior:**

1. **Strip common prefix** — Finds the common prefix of all model identifiers (e.g. `picid.model.forecasters.`) and removes it so names are shorter (e.g. `patchtst_model.PatchTST_Forecaster`).
2. **Append (linear) / (exponential)** — If a model type column exists (e.g. `model.model_type` with values `"linear"` or `"exponential"`), appends ` (linear)` or ` (exponential)` to the model name. Used to distinguish StatisticalBaselineWrapper variants.

**Parameters:**

- **df** — DataFrame with a model target column (and optionally model type).
- **model_target_col** — Column holding the model identifier. Default: `config.COLUMN_CONFIG["model_target"]` (e.g. `"model._target_"`).
- **model_type_col** — Column holding model type. Default: `"model.model_type"`.

**Returns:** A **copy** of `df` with the model target column updated (no new columns).

---

### Example 1: Default columns (use config)

```python
from picid_report.preprocess import clean_and_rename_models

df = clean_and_rename_models(df)
# Uses config.COLUMN_CONFIG["model_target"] and "model.model_type"
# e.g. "picid.model.forecasters.patchtst_model.PatchTST_Forecaster" becomes "patchtst_model.PatchTST_Forecaster"
# If model.model_type is "linear" then "... (linear)" is appended
```

---

### Example 2: Custom column names

```python
df = clean_and_rename_models(
    df,
    model_target_col="model.name",
    model_type_col="model.variant",
)
# Uses your project column names
```

---

### Example 3: No model type column

If `model_type_col` is not in the DataFrame, only the prefix stripping is applied; no (linear) or (exponential) is appended.

```python
# df has no "model.model_type"
df = clean_and_rename_models(df)
# Only prefix stripped
```

---

### Example 4: Skip preprocessing in pipeline

To keep raw model names (no cleaning), pass a no-op preprocess when calling run_pipeline:

```python
from picid_report.run import run_pipeline

df, all_results, summary_df = run_pipeline(
    project_name="my_project",
    user="my_team",
    preprocess_df=lambda df: df,
)
```

---

## 2. Interaction with pipeline

- **run.py** calls `clean_and_rename_models(df)` by default in the preprocess stage. You can override with **preprocess_df** (e.g. custom function or `lambda df: df` to skip).
- Column names come from **config.COLUMN_CONFIG** unless you pass **model_target_col** or **model_type_col**.

See [run.md](run.md) for full pipeline options.

---

**Navigation:** [← Documentation index](README.md)
