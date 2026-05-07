# Docstring audit: known issues and what is wrong

This note describes **documentation problems** introduced or exposed by a broad NumPy-doc pass across `picid/`. It complements the project’s [docstring style guide](docstring_style.md): that guide says *how* to format strings; this document says *what went wrong* when content was replaced by structure alone.

---

## 1. `picid/data/datasources/`: generic filler instead of domain docs

### What happened

Many modules and methods had **specific** explanations (PHMD loading, cache fingerprints, `payload_cache_path`, column mapping, etc.) replaced by:

- Vague module one-liners (e.g. “Provide datasource helpers for phmd loader.”).
- Class summaries that **sound** like datasources but add no contract (e.g. “Load the PHMDMulti Source datasource.” — awkward wording and no behavior).
- Repeated placeholders:
  - `**kwargs : Any, optional` with body **“Description of kwargs.”**
  - `Returns` sections that name type `Any` and description “Result of the operation.”
  - Parameters typed as **`Any`** with a generic line like “Description of `param_name`.”

### Why this is wrong

1. **Loss of API contract.** Callers and maintainers no longer see *what* to pass, *what* is optional, or *how* caching and fingerprints interact. The docstring stops answering “how do I use this safely?”
2. **False precision.** Saying `Returns` → `Any` / “Result of the operation” for a method that returns nothing meaningful, or a known `dict`/`str`, is **misleading**: it looks documented but carries zero information.
3. **Inconsistent with the rest of the codebase.** Outside `datasources/`, many edits add real `Parameters`/`Returns`. Inside `datasources/`, the same *shape* is used to **hide** the old narrative — so readers get a false sense of uniformity.
4. **Pre-commit scope.** `numpydoc-validation` in `.pre-commit-config.yaml` is scoped to `^picid/` **excluding** `^picid/data/datasources/`. Filler in datasources is less likely to be caught by the same checks as the rest of `picid/`.

### Canonical example (illustrative)

File: `picid/data/datasources/base/phmd_loader.py`.

- **Module doc:** A full paragraph describing PHMD multi-source patterns was replaced by a single vague line.
- **`PHMDMultiSourceLoader`:** Detailed `**kwargs` (fold, cache, `payload_cache_path`, fingerprint semantics) was replaced by “Description of kwargs.”
- **Methods** such as `_build_cache_config`, `load_data`, `__repr__`: concrete return descriptions were replaced by “Result of the operation.”

Compare with the last good version:

```bash
git show HEAD:picid/data/datasources/base/phmd_loader.py
```

---

## 2. Information loss outside datasources

### What happened

In several files, **long narrative docstrings** (motivation, edge cases, window semantics, examples) were replaced by **short NumPy-style blocks** that are formally valid but **omit** tutorial and cautionary detail. In some cases the removed text was **operational documentation**: formulas and parameters that explain **dataset length**, **padding**, and **batch shape contracts**. Losing that content makes debugging off-by-one lengths, window counts, and DataLoader behavior much harder—it is not “nice to have” prose.

Affected examples (non-exhaustive):

| File | What was reduced or removed |
|------|-----------------------------|
| `picid/data/datasets/sliding_window_batch_dataset.py` | **See dedicated subsection below** — among the worst losses: full **dataset length math** for dense series, `padding_left_flag` semantics, and `__getitem__` batch contract notes. |
| `picid/data/datasets/fault_classification_dataset.py` | Extended class doc, parameter semantics, **`pred_len` forced to 0**, full **`Examples`** block; some useful inline comments in `__getitem__`. |
| `picid/data/optimization/ak_jagged_sequencer.py` | **See dedicated subsection below** — window index math, padding/`warmup_steps` semantics, **`edge` vs `zero`** modes, worked **Examples**, and **tensor shape** + padding notes on batch methods. |
| `picid/data/preprocessing/preprocessor.py` | **See dedicated subsection below** — full **module tutorial** (framework vs library, runners, protocol), and **API migration** note about removed `mode` / `split_mode`. |
| `picid/transforms/base_transforms/tabularizers.py` | **See dedicated subsection below** — **Hydra YAML comment recipe** removed; **time-features / `freq` narrative** dropped from class doc; **`transform_data`** shape and return specificity reduced. |
| `picid/model/methods/statistical_models.py` | PHM benchmarking context, **equations**, **`Attributes`**, and detailed `config` / `task_type` semantics for baselines. |

#### `picid/data/datasets/sliding_window_batch_dataset.py` (`SlidingWindowBatchDataset`)

This file is called out explicitly because a **large block of technically important content** was removed from the class docstring and replaced by shorter parameter lines that no longer carry the same meaning.

**What the previous documentation provided (and why it mattered):**

1. **Architecture reminder.** One **sequencer per key** in `data_dict`, shared window count across keys, support for **dense and ragged** arrays, and explicit note that **time features are not required** — all of that framed how multi-key batches stay aligned.

2. **Parameter semantics (behavioral, not cosmetic).** The old text explained, for example:
   - What **`padding_left_flag`** actually does: `True` (default) allows windows to start before index 0 with left padding (`min_start = -(seq_len - 1)`); `False` restricts to non-padded windows (`min_start = 0`) and **changes dataset length**.
   - How **`subset_ratio`**, **`subset_seed`**, and **`subset_blocks`** interact with random subset sampling (with a pointer to subset sampling behavior).

3. **Dataset length for a dense time series (removed entirely).** The former docstring included a full subsection **“Dataset length (dense single time series of length T)”** with:
   - `required_len = seq_len + pred_len + pred_offset`
   - Valid start indices `range(min_start, max_start, stride)` and `max_start = T - required_len + 1`
   - The closed form for **`len(ds)`** in terms of `min_start`, `max_start`, and `stride`
   - **Worked numeric examples** for `padding_left_flag=False` vs `True` (e.g. `T=20`, `seq_len=2`, `pred_len=5`, `stride=1` → different window counts)

   That material is **directly useful** when reconciling epoch sizes, debugging “why is my dataset length N?”, or comparing behavior to sequencers elsewhere (e.g. `AkwardJaggedAutoregressiveSequencer`).

4. **`__getitem__` contract.** The one-liner that outputs are **`{key: (x, y)}`** was expanded structurally, but the note that **for drivers without time features the last two entries are `None`** was dropped — relevant for anyone inspecting tuple structure or writing collate logic.

5. **`**kwargs` wording.** Previously described as passed through to the **base class**; the replacement text says **“Reserved for forward compatibility”**, which is a **different semantic claim** and may mislead readers about inheritance behavior.

**Restore reference:**

```bash
git show HEAD:picid/data/datasets/sliding_window_batch_dataset.py
git diff HEAD -- picid/data/datasets/sliding_window_batch_dataset.py
```

#### `picid/data/optimization/ak_jagged_sequencer.py` (`AkwardJaggedAutoregressiveSequencer`)

This sequencer sits on the **critical path** for window boundaries, padding, and multi-unit ragged data. The old class docstring was long because it encoded **invariants** and **indexing contracts** that are easy to get wrong when tuning `seq_len`, `label_len`, `pred_len`, and padding.

**What the previous documentation provided (and why it mattered):**

1. **Problem framing.** Explicit distinction between **multi-unit jagged** data (first dimension = units, variable time per unit) and **dense** single-series arrays — not redundant marketing text; it tells you how `_build_indices` and slicing interpret axes.

2. **“Key features” (behavioral).** Numbered list covering: iteration over units with windows **relative to each unit**; **strict** windowing (short units dropped; input windows are real data only); **left padding / warmup** so prediction can start earlier in the timeline.

3. **Sequence structure (anchor `t`).** Clear slice formulas:
   - **Input `seq_x`:** `[t - seq_len : t]`
   - **Target `seq_y`:** `[t + pred_offset : t + pred_offset + label_len + pred_len]`
   That is the **authoritative** link between constructor arguments and what “a window” means.

4. **Shape documentation for `features`.** Ragged vs dense layouts were spelled out: `(N_Units, var_Time, D_Features)` vs `(Time, D_Features)`.

5. **Parameter semantics removed or flattened.**
   - **`label_len`:** Described as overlap / decoder context and tied to **“decoder start token”** usage in Transformers — the short replacement loses that design intent.
   - **`padding_left_flag`:** Old text distinguished **strict** (`False`/`0`: full `seq_len` history without padding) vs **padding** (`True`/`1`: windows can start before 0, negative indices filled per `padding_mode`).
   - **`padding_mode`:** **`edge`** (repeat first observation) vs **`zero`** was explicit; the replacement only says “strategy for negative indices.”
   - **`warmup_steps`:** The dual meaning was documented — under **`padding_left_flag=True`** (number of padded steps in the first window, default `seq_len - 1` vs `0` for no padding) vs under **`False`** (index offset for the first sequence, default `0`). The replacement (“Explicit warmup offset when provided”) **does not recover that contract**.

6. **Worked examples.** Two scenarios with data `[A, B, C, D]`, `seq_len=3`, stepping through **which windows are produced** under default warmup vs `warmup_steps=0` — the closest thing to a **spec test in prose**.

7. **Batch API: shapes and padding note.** On the method that returns windows for indices, the old doc gave **concrete tensor shapes** (`seq_x`: `(batch, seq_len, num_features)`, `seq_y`: `(batch, label_len + pred_len, num_features)`) and a **Notes** block: with left padding, early **`seq_x`** may contain padded history per `padding_mode`, while **`seq_y`** uses real observations. The new docstring is structurally valid but **drops those contracts**, which hurts anyone validating tensors or debugging padding artifacts.

**Restore reference:**

```bash
git show HEAD:picid/data/optimization/ak_jagged_sequencer.py
git diff HEAD -- picid/data/optimization/ak_jagged_sequencer.py
```

#### `picid/data/preprocessing/preprocessor.py` (`PreProcessor` module)

Here the loss is less about formulas and more about **entry-point documentation** and a **breaking/API note** that used to live on the class.

**What the previous documentation provided (and why it mattered):**

1. **Module-level tutorial (removed).** The old module docstring was effectively a **mini-guide**:
   - **Framework path** (`run.py`): instantiate `ConfigTransformManager`, `PreProcessor`, call `pipeline(cache_paths=..., cache_preprocessed=True)` with a concrete sketch.
   - **Library path**: import `TransformPipeline`, build a list of `DataTransform` steps, run with or without the same **three-tier caching** as the framework by passing `PreProcessor(..., transforms=pipeline)` and `pipeline(..., data_cache_path=..., transform_library_part_path=..., cache_preprocessed=True)`.
   - Explicit statement that **`ConfigTransformManager` and `TransformPipeline` both satisfy the same protocol** and that **`PreProcessor` / runners depend only on that protocol** — central to understanding why both styles exist.

2. **Pipeline runners (removed).** Named **`DirectPipelineRunner`** vs **`CachingPipelineRunner`** and described the caching tiers (`loaded_and_splitted_data` → boundary → preprocessed) and **optional boundary restore**. The replacement paragraph says behavior is easier to test but **no longer points readers** to those runner names or cache stages when debugging cache misses.

3. **Migration / configuration gotcha (removed from class doc).** The old `PreProcessor` class doc stated that the **`mode` parameter (`"per_unit"` / `"cross_unit"`) was removed** and told callers to obtain split mode from the datasource when needed, e.g. `split_mode = datasource.get_split_mode()`. That is **actionable changelog-style documentation**; removing it risks repeated confusion or wrong assumptions about constructor arguments.

4. **Smaller but real losses in method docs.** Examples: `after_each_transform_callback` used to note it is **used internally for boundary cache saves**; `_extract_log_string` used to document **preference order** (“train”, then “test”, then first key). Generic replacements hide **operational detail** useful when reading logs or stack traces.

**Restore reference:**

```bash
git show HEAD:picid/data/preprocessing/preprocessor.py
git diff HEAD -- picid/data/preprocessing/preprocessor.py
```

#### `picid/transforms/base_transforms/tabularizers.py` (`TimeseriesTabularizer`)

**What was removed (and why it matters):**

1. **Hydra-oriented comment block at the top of the file (removed entirely).** The old file led with a **copy-pastable YAML sketch** for a `tabularize` transform: `_target_` → `TimeseriesTabularizer`, example **`select_features`** entries (`time_features` / `features` / `target` with modes like `t` and `history`), and **`metadata.apply_to`** / **`assign_to`**. That is not redundant noise—it is **wiring documentation** for configs. Replacing it with a one-line module docstring loses the **concrete mapping** from config keys to transform behavior.

2. **Class doc: purpose vs parameters (editorial tradeoff, with a real loss).** The previous class docstring described the transform as **extracting time-based features** via the **`time_features` utility** and documented **`freq`** (e.g. hourly vs daily) plus `**kwargs` for the base transform. The **Parameters** section was **out of sync** with `__init__`: the constructor already took `select_features`, `seq_len`, `label_len`, etc., while the doc only mentioned `freq`. The new class doc **correctly lists** the main constructor arguments—but it **drops the time-feature story** and any explicit mention of **`freq`**. Because `freq` is still a natural pass-through via `**kwargs` (see the remaining `__repr__` comment about including `freq`), readers lose **documented guidance** on that important knob unless they infer it from elsewhere.

3. **`transform_data` contract watered down.** The old docstring stated an **expected input layout**—data shaped **`(N, T, ...)`** with **`N`** = samples and **`T`** along the time/task axis (wording in the original was easy to misread but still anchored expectations). It also described the return as **NumPy** output of **extracted time features**. The replacement uses generic **`Any`** for the return and drops the **shape hint**, which makes it harder to validate outputs or relate this transform to the **sequencing / tabularization** pipeline.

4. **Inline comments under `__init__` unchanged.** The diff keeps the “pass kwargs to `BaseTransform`” comments; the issue here is specifically **user-facing** doc loss (YAML recipe + class/transform_data narrative), not a logic change.

**Restore reference:**

```bash
git show HEAD:picid/transforms/base_transforms/tabularizers.py
git diff HEAD -- picid/transforms/base_transforms/tabularizers.py
```

### Why this is wrong

1. **Edge cases and invariants disappear.** Uniform short docs rarely repeat “when padding applies”, “what happens for short units”, or “which kwargs are ignored” — exactly where bugs come from. When **explicit length formulas and numeric examples** are removed (as in `sliding_window_batch_dataset.py`), maintainers lose a **checkable specification** of `len(dataset)` and window indexing.
2. **Onboarding cost goes up.** New contributors lose the “why this class exists”, runnable examples, and **copy-pastable entry-point recipes** (e.g. the old `preprocessor.py` module doc) without gaining much from repeated boilerplate.
3. **Not the same problem as §1.** Here the text is often **grammatically fine**; the issue is **editorial**: trading depth for consistency. That may be acceptable per class if reviewed deliberately — but mass replacement is risky.

---

## 3. Doc vs types, and redundant structure

### 3a. Return type hints disagree with docstrings and behavior

File: `picid/pipeline/base.py`.

`BackboneWrapperLightningModule._validation_step` and `_test_step` are annotated **`-> None`** but the docstrings state they **`Returns` a `dict`**, and the body **`return model_out`**. Meanwhile, `CustomEvaluatorInterface` documents similar steps with **`-> dict[str, torch.Tensor]`**.

**Why this is wrong**

- **Type checkers and readers** trust annotations first; **`None` vs dict** is a direct contradiction.
- Docstrings that describe a **`dict`** return while the signature says **`None`** look like a **partial migration** (docs updated, types not).

**Fix direction (conceptual):** align `->` with reality (and Lightning expectations), or adjust docs if the API is intentionally untyped/`None` for a reason.

### 3b. `Parameters` on exception classes duplicate `__init__`

File: `picid/exceptions.py`.

Classes such as `TransformError` and `PreprocessingDatasourceError` have a full **`Parameters`** block on the **class** docstring, then the **same** parameters repeated on **`__init__`**.

**Why this is wrong**

- NumPy style usually documents **constructor arguments on `__init__`** (or the class doc if there is no `__init__` doc). Duplication **doubles maintenance** and can drift.
- It is **unidiomatic** for exceptions compared to common NumPy/Sphinx patterns.

### 3c. Repeated low-signal metric docstrings

File: `picid/metrics/metrics.py`.

Many metric classes repeat nearly identical **`reset`**, **`update`**, and **`compute`** docstrings (“Update the metric with a new batch…”) with only the implementing code differing.

**Why this is wrong**

- Readers skim repeated blocks and **stop trusting** that anything important changes between classes.
- For shared behavior, a **base-class doc** or a one-line subclass override is often clearer than copying a full **`Parameters`** block everywhere.

---

## Summary

| Issue | Severity | Core problem |
|-------|----------|----------------|
| Datasources filler | **High** | Structure without content; misleading `Any` / “Result of the operation.” |
| Narrative trimmed elsewhere | **Medium** (higher where math/contracts removed) | Valid NumPy shape; lost edge cases, examples, **operational specs** (`sliding_window_batch_dataset.py`, `ak_jagged_sequencer.py`), **module-level guides** (`preprocessor.py`), and **config recipes** (`tabularizers.py`). |
| `-> None` vs `Returns dict` | **Low–medium** | Annotations and docs disagree with implementation. |
| Exception / metric duplication | **Low** | Redundant or generic text, harder to maintain. |

---

## Related commands

```bash
# Inspect datasource regression vs last commit
git diff HEAD -- picid/data/datasources/base/phmd_loader.py

# Restore a path to committed version (destructive to working tree — use with care)
# git checkout HEAD -- picid/data/datasources/<file>
```

When improving docs, prefer **restoring accurate domain text** and then applying [NumPy structure](docstring_style.md), rather than replacing explanations with placeholders.
