# Developer-Level Introduction: Internals and Extension Points

This guide is for contributors implementing new transforms or modifying core behavior.

---

## Core Interfaces

### `BaseTransform`

Located in `lmetk/transforms/base/base_transform.py`.

```python
class BaseTransform(ABC):
    def fit_data(self, data, metadata): ...
    def transform_data(self, data, metadata): ...
```

Only `transform_data` is required. `fit_data` is optional for stateless transforms.

Marker classes define supported data types:

- `DenseTransform`
- `RaggedTransform`
- `RaggedOrDenseTransform`

These markers are used by the multi-source machinery to decide how to route ragged data.

---

## Orchestration Flow (Call Graph)

The high-level call chain is:

1. `ConfigTransformManager` builds `DataTransform` objects
2. `DataTransform.forward(...)` delegates to `TransformStrategy.apply(...)`
3. `TransformStrategy` handles:
   - hybrid copy semantics (deep for NumPy, shallow for Awkward)
   - optional fitting (`fit_on`, `fit_on_key`)
   - per-split transform (respecting `transform_on_keys`)
   - merge into `assign_to`

When multi-unit, `TransformStrategy` provides a list of `NamedTransformInput` chunks.
Each chunk contains the subset of keys listed in `apply_to`.

---

## Metadata Parsing and Validation

`DataTransform` performs:

- `apply_to` normalization (string vs list)
- `assign_to` mapping expansion (string vs list vs mapping)
- `fit_on` / `fit_on_key` checks
- signature validation for `transform_data(...)`

If `apply_to` is a list and `fit_on` is set, `fit_on_key` is required to avoid ambiguity.

---

## Signature Expectations (And a Caveat)

`DataTransform` validates the `transform_data` signature. The default accepted form is:

```python
def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
    ...
```

If `apply_to` is a mapping (e.g. `apply_to: {img: images, lbl: targets}`),
the validator accepts named parameters; however, **execution currently does not
support mapped `apply_to` in `DataTransform.forward()`**, so this should be avoided
unless you are extending core logic.

---

## Multi-Source Mixins (Internal Behavior)

All multi-unit behavior lives in `lmetk/transforms/base/multisource.py`. Prefer the current names:

- **`StandardFitTransformMixin`** (alias: `ConcatinatedFitMultiSourceTransformMixin`)
  - Fit: concatenate all sources, then fit once
  - Transform: apply per segment
  - Ragged handling: dense-only (flatten → NumPy → reassemble) or ragged-native (Awkward directly)

- **`StatelessPerSegmentMixin`** (alias: `NoFitMultiSourceTransformMixin`)
  - No fit; transform applied per segment. Raises if `fit_multi_source` is called.

- **`StatelessConcatenateMixin`** (alias: `ConcatenateUnitsMixing`)
  - No fit; concatenates all sources along a required `axis`, then one `transform_data` call.

- **`InverseTransformMixin`**
  - Implement abstract `inverse_transform`; mixin provides `inverse_transform_data` and `inverse_transform_multi_source`.

---

## Output Normalization

`TransformStrategy` expects one of:

- a single array / object
- a list of arrays (one per unit)
- a list of dict-like objects

These are normalized by `postprocess_transformed_data(...)` into return objects
that map to `assign_to_map`. If your transform returns a dict, ensure the keys
match the mapping in `metadata.assign_to`.

---

## Ragged Data Internals

Key functions involved in ragged handling:

- `find_singular_ragged_dim(...)`
- `ak_regularize_regular_axes(...)`
- `ragged_index_tuples(...)`

Dense-only transforms on ragged data trigger a slow path:

1. Iterate every ragged sub-segment
2. Convert to NumPy
3. Apply transform
4. Reconstruct the Awkward structure

If performance matters, implement ragged-native logic and inherit from `RaggedTransform`
or `RaggedOrDenseTransform`.

---

## Debugging and Transform Logs

`transform_multi_source(...)` returns a `transform_log` with flags such as:

- `data_is_ragged`
- `data_is_ragged_but_regular`
- `transform_supports_ragged`
- `transform_supports_dense`

These are useful for understanding why a transform took a slow path.

---

## Common Extension Patterns

- **Pure stateless transform:** `BaseTransform` + `StatelessPerSegmentMixin`
- **Global-fit scalers:** `BaseTransform` + `StandardFitTransformMixin`
- **Unit-merge transforms:** `BaseTransform` + `StatelessConcatenateMixin` (must set `axis` in `__init__`)
- **Inverse projections:** `InverseTransformMixin` + implement `inverse_transform`
