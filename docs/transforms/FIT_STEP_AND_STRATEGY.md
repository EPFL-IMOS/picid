# FitStep: `fit_on_split` / `fit_on_key` and `context.strategy`

## 1. What is `context.strategy` and how is it set?

**`context.strategy`** is the **TransformStrategy** instance that is running the pipeline. It is set in exactly one place: when the strategy builds the context inside its own `apply()` method:

```python
# lmetk/transforms/base/strategy.py (around line 96)
context = TransformContext(
    ...
    strategy=self,   # <-- self is the TransformStrategy
    postprocess_fn=postprocess_transformed_data,
    ...
)
```

So **whoever calls `TransformStrategy().apply(...)`** is the same object that ends up in `context.strategy`. The strategy is attached to the context so that pipeline steps can call back into it:

- **FitStep** uses `strategy._prepare_chunks_for_split(...)` to get the right chunks for the fit split (e.g. train) and key.
- **TransformStep** uses `strategy._prepare_chunks_for_split(...)` to prepare chunks per split for transformation.
- **MergeStep** uses `strategy._merge_transformed_data(...)` to merge transformed results into `transformed_data`.

So the strategy is the **orchestrator**: it owns the pipeline and the helpers for chunk preparation and merging. The context is the shared state passed through the steps; steps use `context.strategy` when they need to call those helpers. Without `context.strategy`, FitStep (and others) would not know how to prepare chunks for the container’s structure (single vs multi-unit, etc.).

---

## 2. Why `if not context.fit_on_split or not context.fit_on_key: return context`?

The intent is to make **FitStep generic** for all transforms:

- **Transforms that do not need fitting**  
  Metadata does not set `fit_on` / `fit_on_key` (so `fit_on_split` and `fit_on_key` are `None`). FitStep should do nothing and not call any fit method. The condition `if not context.fit_on_split or not context.fit_on_key: return context` achieves that: we exit early and skip fitting.

- **Transforms that need fitting**  
  Metadata sets both `fit_on_split` and `fit_on_key`. The condition is false, so we proceed and run the fit (using `context.strategy._prepare_chunks_for_split`, then `fit_multi_source` or `fit_data`).

So the current logic is: **“Only fit when both are set; otherwise skip.”** That avoids calling fit on stateless transforms and keeps one step that works for both fittable and non-fittable transforms.

---

## 3. The problem: user error when a transform requires fitting

If a transform **does** require fitting (e.g. a scaler) but the user forgets to set `fit_on` / `fit_on_key` in metadata:

- With the current logic we **skip** fitting (because `fit_on_split` or `fit_on_key` is missing).
- The transform is then used unfitted and may fail later (e.g. in TransformStep) with a confusing error like “Transform must be fitted first” or an attribute error.

So we want: **when the transform requires fitting but `fit_on_split` or `fit_on_key` is missing, raise a clear error** that explains what is wrong and how to fix it (e.g. set `fit_on` and optionally `fit_on_key` in metadata).

---

## 4. How to know if a transform “requires fitting”

Two practical options:

**Option A – Heuristic: “has fit capability”**  
Consider the transform as requiring fitting if it **overrides** the default no-op `fit_data` or implements `fit_multi_source` in a non-stub way. For example:

- Check that `fit_data` is overridden (e.g. the implementation is not `BaseTransform.fit_data`).
- Or check that the class has a custom `fit_multi_source` that does not only raise `NotImplementedError`.

**Option B – Explicit flag (implemented)**  
`ConcatinatedFitMultiSourceTransformMixin` sets `requires_fit = True`; `NoFitMultiSourceTransformMixin` sets `requires_fit = False`. FitStep does:

```text
requires_fit = getattr(context.transform_instance, "requires_fit", False)
if not context.fit_on_split or not context.fit_on_key:
    if requires_fit:
        raise ValueError("Transform ... requires fitting but fit_on_split and/or fit_on_key are not set. ...")
    return context
```

---

## 5. Proposed behavior for FitStep (summary)

Goal:

1. **No fitting needed** (stateless transform): if `fit_on_split` or `fit_on_key` is missing, do nothing. (Current behavior.)
2. **Fitting needed** (fittable transform): if either is missing, **raise a clear ValueError** with message and fix suggestion instead of silently skipping.
3. **Both set**: run fit as today (using `context.strategy._prepare_chunks_for_split`, then `fit_multi_source` or `fit_data`).

Proposed logic (pseudocode):

```text
if not context.fit_on_split or not context.fit_on_key:
    if transform_requires_fitting(context.transform_instance):
        raise ValueError(
            "Transform ... requires fitting but fit_on_split and/or fit_on_key are not set. "
            "Set metadata 'fit_on' (e.g. 'train') and optionally 'fit_on_key' (defaults to apply_to)."
        )
    return context
# else: both set, run fit as now
strategy = context.strategy
...
```

Detection of “transform requires fitting” can be:

- **Phase 1:** Use a heuristic (e.g. `fit_data` overridden and not the base no-op, or has a real `fit_multi_source`). Document the convention.
- **Phase 2 (optional):** Introduce `requires_fit = True` on fittable transforms/mixins and use that instead for a more stable contract.

---

## 6. Where to implement

- **FitStep** in `lmetk/transforms/base/pipeline.py`: add the “requires fitting but missing fit_on_*” check and the `ValueError` with message and fix suggestion.
- **Helper** “does this transform require fitting?” can live in the same module or in a small helper in `pipeline` or `integrity` to keep the step readable.

No change to **what** `context.strategy` is or **how** it is set; only the FitStep condition and error handling are extended so user mistakes for fittable transforms are caught early with a clear message.
