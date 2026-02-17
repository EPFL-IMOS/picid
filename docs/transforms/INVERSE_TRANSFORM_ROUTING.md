# Inverse Transform Routing — Evaluation & Plan

**Context:** Multiple scalers in the pipeline (e.g. `scaler_features`, `scaler_target`, `final_scaler_features`). The evaluator usually needs the **target scaler** to inverse-scale predictions/targets; sometimes it needs a **feature scaler** or a **specific** transform by name.

---

## 1. How the pipeline maps to keys

Transforms are configured with:

- **apply_to:** key(s) read from (e.g. `features`, `target`)
- **assign_to:** key(s) written to (defaults to apply_to when omitted)

The **data key** that holds the *output* of a transform is **assign_to**. That is what `get_inverter_for_key(key)` matches on.

| Transform (example)      | apply_to | assign_to | Data key for inverse lookup |
|--------------------------|----------|-----------|-----------------------------|
| scaler_features          | features | (features)| **features**                |
| scaler_target (unibo)    | target   | **rul**   | **rul**                     |
| scaler_target (pronostia)| target, unit_id | **rul** | **rul**                     |
| final_scaler_features    | features | (features)| **features**                |

So:

- For **battery/unibo**: the scaled target lives under the key **rul**. The transform that produced it is `scaler_target` (assign_to: rul).
- For **features**: both `scaler_features` and `final_scaler_features` write to **features**. By key you get one of them (default: **last** in pipeline order).

---

## 2. What is already implemented

- **`get_inverter_for_key(key, which="last"|"first")`**  
  Returns the transform (InverseTransformMixin) that **assigns to** `key`. When several match, **which="last"** (default) returns the last in pipeline order; **which="first"** returns the first.

- **Evaluator config (run.py)**  
  - **inverse_transform_key:** e.g. `rul` or `features`. Resolved via `get_inverter_for_key(inverse_transform_key)`. No transform name needed.  
  - **inverse_transform_name:** e.g. `scaler_target`. Resolved via `get_transform(name).transform_instance`. Explicit name.  
  - If `apply_inverse_scaling` is True, key is tried first; if unset or no match, name is used. If neither is provided, an error is raised.

- **Standard case (evaluator expects “scaler_target”)**  
  - **By key:** Set `inverse_transform_key: rul` (because scaler_target has `assign_to: rul`). The manager returns the transform that writes to `rul`, i.e. scaler_target.  
  - **By name (current configs):** Keep `inverse_transform_name: scaler_target`. Behaviour unchanged.

- **Other scalers (e.g. feature scalers in evaluator)**  
  - **By name:** Use `inverse_transform_name: scaler_features` or `inverse_transform_name: final_scaler_features` when you need a **specific** transform.  
  - **By key:** Use `inverse_transform_key: features` when you want “whoever last wrote to features” (typically `final_scaler_features`). For “first” writer to features, the API supports `which="first"` in code; evaluator config does not expose `which` yet.

---

## 3. Gaps and possible confusion

1. **Key vs name in docs**  
   The report previously said “e.g. inverse_transform_key: target”. For unibo/pronostia the correct key is **rul** (assign_to), not `target` (apply_to). Docs and examples should say: **inverse_transform_key is the data key (assign_to)** where the scaled values live, e.g. `rul` or `features`.

2. **Config examples**  
   Default evaluator configs only show `inverse_transform_name`. They do not show `inverse_transform_key` as the alternative for the standard “target scaler” case.

3. **Exposing `which` in config (optional)**  
   If we want “first feature scaler” from config (e.g. `inverse_transform_key: features`, `inverse_transform_which: first`), run.py would need to pass `which` into `get_inverter_for_key`. Low priority.

4. **Naming convention**  
   Transform names (`scaler_target`, `scaler_rul`, `scaler_features`) are project-specific. Key-based lookup is stable: same key across projects (e.g. `rul`, `features`).

---

## 4. Plan — steps to finish and clarify

### 4.1 Documentation (high priority)

- [x] **Developer / user doc**  
  - State clearly: **inverse_transform_key** = data key (assign_to), e.g. `rul` for the scaled target when assign_to is rul, `features` for feature scalers.  
  - State: **inverse_transform_name** = transform config name; use when you need a specific transform (e.g. scaler_features vs final_scaler_features).  
  - Add one “Standard case” example: “To pass the target scaler to the evaluator, set inverse_transform_key: rul (if your scaler has assign_to: rul).”  
  - Add one “Multiple scalers” example: “If several transforms assign to the same key, the last in pipeline order is used unless you use inverse_transform_name to pick one.”

- [x] **Config examples**  
  - In a default or example evaluator config (or in this doc), add a commented example:  
    `# inverse_transform_key: rul   # use key when assign_to is rul; then inverse_transform_name not needed`

- [x] **PHASE0 report**  
  - In §9, change the example from `inverse_transform_key: "target"` to `inverse_transform_key: "rul"` (or “e.g. rul when assign_to is rul”) so it matches real configs.

### 4.2 Optional enhancements (lower priority)

- [x] **Evaluator config: inverse_transform_which**  
  - If we want “first” vs “last” from config (e.g. for features), add `inverse_transform_which: first|last` and pass it to `get_inverter_for_key(key, which=...)`. Default `last`.

- [x] **Validation / logging**  
  - Done: run.py logs chosen transform when using key-based lookup (e.g. “Using transform ‘scaler_target’ for inverse_transform_key=rul”). Helps when multiple scalers exist.

### 4.3 No change required

- **run.py resolution order** (key first, then name) is correct.  
- **get_inverter_for_key(key, which)** semantics (assign_to, pipeline order) match the pipeline.  
- **Existing configs** that use only `inverse_transform_name: scaler_target` (or scaler_rul, scaler_features) keep working.

---

## 5. Summary

| Goal | Recommended config | Notes |
|------|--------------------|--------|
| Evaluator gets target scaler (standard) | `inverse_transform_key: rul` | Key = assign_to; works when scaler has assign_to: rul. |
| Same, by transform name | `inverse_transform_name: scaler_target` | Current pattern in most configs; keep as is. |
| Evaluator gets a specific feature scaler | `inverse_transform_name: scaler_features` or `final_scaler_features` | Use name when multiple transforms write to same key. |
| Evaluator gets “last” feature scaler | `inverse_transform_key: features` | No name needed; last in pipeline order. |

Implementation is aligned with the pipeline; remaining work is documentation and optional config/validation improvements.
