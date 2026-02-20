# kwargs.pop() → Explicit Signatures Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all `kwargs.pop()` usage in the TabPHM codebase with explicit function parameters so that IDEs, type checkers, and readers see clear signatures.

**Architecture:** Apply the refactor in layers: (1) data objects and datasets, (2) model wrappers. Use the patterns from the strategy doc: Required Argument, Default Value, Legacy Alias (with deprecation warning where applicable), and keep `**kwargs` where needed for `super().__init__(**kwargs)` or Hydra. No behavioral change; only signature and extraction logic.

**Tech Stack:** Python 3.x, type hints, existing tests (pytest). Hydra configs pass named keys that will map to new explicit parameters.

**Out of scope:** `local_packages/TabPFN` (external dependency); any `**kwargs` that do not use `pop()`.

---

## Affected files (summary)

| File | Keys popped | Pattern |
|------|-------------|--------|
| `picid/data/data_objects/base.py` | `metadata` | Default value |
| `picid/data/data_objects/data.py` | `manifest`, `slice_info` | Default value |
| `picid/data/datasets/fit_predict_dataset.py` | `dataset_cfg` | Default value (optional) |
| `picid/model/wrappers/mlp_wrapper.py` | `num_targets`/`output_dim`/`num_classes`, `input_channels`, `hidden_dim`, `num_layers` | Required + default + legacy alias |
| `picid/model/wrappers/statistical_models_wrapper.py` | same + `poly_degree` | Same |
| `picid/model/wrappers/cnn1d_wrapper.py` | `num_classes`, config keys, `cls_head` | Required/optional by task + explicit config params |
| `test/data/datasources/base/conftest.py` | `data_splitter` | Default value |

---

## Task 1: BaseDataObjectWithMetadata — explicit `metadata`

**Files:**
- Modify: `picid/data/data_objects/base.py` (lines 232–247)
- Test: `test/data/data_objects/test_base_and_manifest_coverage.py` (and any test that builds `BaseDataObjectWithMetadata` via subclasses)

**Step 1: Add type import if missing**

Ensure `Optional` and `Dict`/any needed types are imported in `base.py`.

**Step 2: Update signature and remove pop**

In `picid/data/data_objects/base.py`, change:

```python
def __init__(self, **kwargs: Any) -> None:
    ...
    self._metadata: Optional[Dict[str, Any]] = None
    metadata = kwargs.pop("metadata", None)
    super().__init__(**kwargs)
```

To:

```python
def __init__(self, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
    self._metadata: Optional[Dict[str, Any]] = None
    super().__init__(**kwargs)
    if metadata is not None:
        ...
```

Keep the rest of the method body (setting `_metadata`, column_map logic) unchanged; only the signature and the removal of `metadata = kwargs.pop("metadata", None)` change. Pass `metadata` into the initializer where it’s used.

**Step 3: Run tests**

```bash
pytest test/data/data_objects/ -v
```

Expected: all pass. Fix any call sites that construct `BaseDataObjectWithMetadata(**dict_with_metadata)` to pass `metadata=...` explicitly if needed (or rely on keyword from dict).

**Step 4: Commit**

```bash
git add picid/data/data_objects/base.py
git commit -m "refactor(data): BaseDataObjectWithMetadata explicit metadata param"
```

---

## Task 2: DatasetContainer — explicit `manifest` and `slice_info`

**Files:**
- Modify: `picid/data/data_objects/data.py` (lines 129–139)
- Test: `test/data/data_objects/test_data_coverage.py`, `test/data/data_objects/test_data_objects_characterization.py`

**Step 1: Confirm imports**

Ensure `MetadataManifest` and `SliceInfo` are imported in `data.py`.

**Step 2: Update signature**

Change:

```python
def __init__(self, *args, is_multi_unit: bool = True, **kwargs: Any) -> None:
    ...
    manifest = kwargs.pop("manifest", MetadataManifest())
    self.manifest: Optional[MetadataManifest] = manifest
    slice_info = kwargs.pop("slice_info", None)
    self.slice_info: Optional[SliceInfo] = slice_info
```

To:

```python
def __init__(
    self,
    *args,
    is_multi_unit: bool = True,
    manifest: Optional[MetadataManifest] = None,
    slice_info: Optional[SliceInfo] = None,
    **kwargs: Any,
) -> None:
    self.is_multi_unit = is_multi_unit
    self.manifest = manifest if manifest is not None else MetadataManifest()
    self.slice_info = slice_info
```

Then remove the two `kwargs.pop` lines. Keep the rest of the method (normalization loop, `super().__init__(**kwargs)`) unchanged.

**Step 3: Run tests**

```bash
pytest test/data/data_objects/ -v
```

Expected: all pass. Call sites that pass `manifest=` or `slice_info=` in kwargs continue to work by passing them as named args.

**Step 4: Commit**

```bash
git add picid/data/data_objects/data.py
git commit -m "refactor(data): DatasetContainer explicit manifest and slice_info params"
```

---

## Task 3: FitPredictTaskDataset — explicit `dataset_cfg`

**Files:**
- Modify: `picid/data/datasets/fit_predict_dataset.py` (lines 53–91)
- Test: `test/data/datasets/test_fit_predict_dataset.py`

**Step 1: Update signature and body**

Add optional parameter `dataset_cfg: Optional[dict] = None` to `__init__`. Replace:

```python
if kwargs.get("dataset_cfg", None) is not None:
    dataset_cfg = kwargs.pop("dataset_cfg")
    if dataset_cfg.get("get_unit_id", None):
        self.get_unit_id = True
```

With:

```python
if dataset_cfg is not None and dataset_cfg.get("get_unit_id", None):
    self.get_unit_id = True
```

Add `dataset_cfg: Optional[dict] = None` to the signature (before `**kwargs`).

**Step 2: Run tests**

```bash
pytest test/data/datasets/test_fit_predict_dataset.py -v
```

Expected: all pass. Hydra and tests that pass `dataset_cfg=` remain valid.

**Step 3: Commit**

```bash
git add picid/data/datasets/fit_predict_dataset.py
git commit -m "refactor(datasets): FitPredictTaskDataset explicit dataset_cfg param"
```

---

## Task 4: MLPWrapper — explicit params and legacy alias

**Files:**
- Modify: `picid/model/wrappers/mlp_wrapper.py` (lines 17–78)
- Test: `test/baselines/` (if any MLP tests), and run a short experiment that uses MLP (e.g. from configs)

**Step 1: Add warnings import**

`import warnings` at top if not present.

**Step 2: New signature and extraction**

Target signature (keep `**kwargs` for super):

- Required (by task): for regression `num_targets` or legacy `output_dim` (optional, default 1); for classification `num_classes` (required).
- Required: `input_channels`.
- Optional with defaults: `hidden_dim: int = 64`, `num_layers: int = 2`.

Use **Legacy Alias**: accept `num_classes` as primary; if `output_dim` in kwargs, warn and use it for regression target_dim. Do not pop from kwargs until after resolving alias.

Example shape:

```python
def __init__(
    self,
    task_type: str,
    seq_len: int,
    *,
    input_channels: int,
    num_targets: Optional[int] = None,
    num_classes: Optional[int] = None,
    output_dim: Optional[int] = None,  # deprecated alias for regression
    hidden_dim: int = 64,
    num_layers: int = 2,
    **kwargs,
):
```

Then in body: if regression, prefer `num_targets`, else if `output_dim` in kwargs emit DeprecationWarning and use it; default 1. If classification, require `num_classes` (or from kwargs for backward compat once). Remove all `kwargs.pop` for these keys. Pass remaining `**kwargs` to `super().__init__(**kwargs)`.

**Step 3: Run tests and config**

```bash
pytest test/baselines/ -v -k mlp
# Optional: run one experiment using MLP config
```

Expected: tests pass; configs (e.g. `configs/model/mlp.yaml`) still work via Hydra.

**Step 4: Commit**

```bash
git add picid/model/wrappers/mlp_wrapper.py
git commit -m "refactor(model): MLPWrapper explicit params and output_dim deprecation"
```

---

## Task 5: StatisticalBaselineWrapper — explicit params and legacy alias

**Files:**
- Modify: `picid/model/wrappers/statistical_models_wrapper.py` (lines 28–112)
- Test: `test/baselines/test_statistical_models_wrapper.py`

**Step 1: Same pattern as MLPWrapper**

Explicit parameters: `input_channels` (required), `num_targets` / `num_classes` / `output_dim` (legacy), `poly_degree: int = 2`. Use the same Legacy Alias pattern for regression (`num_targets` preferred, then `output_dim` with deprecation). Remove all `kwargs.pop` for these; keep `**kwargs` for super.

**Step 2: Run tests**

```bash
pytest test/baselines/test_statistical_models_wrapper.py -v
```

Expected: all pass.

**Step 3: Commit**

```bash
git add picid/model/wrappers/statistical_models_wrapper.py
git commit -m "refactor(model): StatisticalBaselineWrapper explicit params and output_dim deprecation"
```

---

## Task 6: CNN1D_Wrapper — explicit num_classes and config keys

**Files:**
- Modify: `picid/model/wrappers/cnn1d_wrapper.py` (lines 37–111)
- Config: `configs/model/cnn_1d.yaml` (no change needed; Hydra passes same keys)

**Step 1: Explicit required config + optional num_classes**

For regression, `num_classes` is not used (effectively 1); for classification, require `num_classes`. Add explicit parameters for all config keys: `input_channels`, `latent_dim`, `dropout_prob`, `output_channels`, `kernels`, `strides`, `dilations`. Optionally add `num_classes: Optional[int] = None` (required when task is classification). Remove the loop that pops from kwargs; instead take these from the signature. Keep `kwargs.pop("cls_head", None)` as a no-op for backward compat or replace with a single explicit `cls_head: Optional[Any] = None` and do not pass to backbone.

**Step 2: Signature shape**

```python
def __init__(
    self,
    task_type: str,
    seq_len: int,
    *,
    input_channels: int,
    latent_dim: int,
    dropout_prob: float,
    output_channels: int,
    kernels: ...,
    strides: ...,
    dilations: ...,
    num_classes: Optional[int] = None,
    cls_head: Optional[Any] = None,
    **kwargs,
):
```

Regression: ignore `num_classes` (or require None). Classification: require `num_classes` is not None. Build `config` from the explicit args; do not pop from kwargs. Then `super().__init__(backbone=backbone, **kwargs)`.

**Step 3: Run tests**

```bash
pytest test/baselines/ -v -k cnn
```

Expected: all pass.

**Step 4: Commit**

```bash
git add picid/model/wrappers/cnn1d_wrapper.py
git commit -m "refactor(model): CNN1D_Wrapper explicit config and num_classes params"
```

---

## Task 7: InMemorySingleSourceLoader (test conftest) — explicit `data_splitter`

**Files:**
- Modify: `test/data/datasources/base/conftest.py` (lines 13–30)

**Step 1: Update signature**

Change:

```python
def __init__(self, n_samples=200, n_features=5, data_name="in_memory", **kwargs):
    ...
    data_splitter = kwargs.pop("data_splitter", None)
```

To:

```python
def __init__(
    self,
    n_samples=200,
    n_features=5,
    data_name="in_memory",
    data_splitter=None,
    **kwargs,
):
```

Set `self._data_splitter` (or use local variable) from `data_splitter`; if `data_splitter is None`, create the default `TimeSplitter(...)` as before. Remove the pop line. Pass `data_splitter=...` into `super().__init__(...)`.

**Step 2: Run tests**

```bash
pytest test/data/datasources/base/ -v
```

Expected: all pass.

**Step 3: Commit**

```bash
git add test/data/datasources/base/conftest.py
git commit -m "refactor(tests): InMemorySingleSourceLoader explicit data_splitter param"
```

---

## Task 8: Full regression and type check

**Step 1: Run full test suite**

```bash
pytest -v
```

Expected: no failures related to the refactored code.

**Step 2: Type check (if project uses mypy/pyright)**

```bash
mypy picid/  # or pyright
```

Fix any new errors in refactored files.

**Step 3: Commit (if any fixes)**

```bash
git add ...
git commit -m "chore: fix tests/typing after kwargs.pop refactor"
```

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-02-20-kwargs-pop-refactor-explicit-signatures.md`.

Two execution options:

1. **Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, and iterate quickly.
2. **Parallel session (separate)** — You open a new session in the same (or a dedicated) worktree and use **superpowers:executing-plans** for batch execution with checkpoints.

Which approach do you prefer?
