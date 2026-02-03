# API Reference

This section documents the public API of core TabPHM modules. Additional packages (evaluators, metrics, pipeline, transforms) are documented in their respective guides; this page focuses on the **data objects** layer used across datasources, transforms, and datasets.

---

## Data objects (`lmetk.data.data_objects`)

The data objects module provides type-safe, dot-notation containers for transporting data through the pipeline. Implementations live in `lmetk.data.data_objects.base`, `lmetk.data.data_objects.data`, and supporting utilities in `lmetk.data.data_objects.utils`. **`NamedTransformInput`** is the standard type used as input to transforms; **`SplitDatasetContainer`** is what datasources return and what the rest of the pipeline (datamodules, datasets) consumes.

### Base classes and mixins

::: lmetk.data.data_objects.base.ToNumpyMixin
::: lmetk.data.data_objects.base.ToDataFrameMixin
::: lmetk.data.data_objects.base.BaseDataObject

### Metadata and dataset containers

::: lmetk.data.data_objects.base.BaseDataObjectWithMetadata

### Transform and dataset container types

::: lmetk.data.data_objects.data.NamedTransformInput
::: lmetk.data.data_objects.data.DatasetContainer
::: lmetk.data.data_objects.data.SplitDatasetContainer

### Return object types (generic)

These types are provided for structured transform outputs; not all are used in every pipeline.

::: lmetk.data.data_objects.data.ReturnObject
::: lmetk.data.data_objects.data.SimpleReturnObject
::: lmetk.data.data_objects.data.NamedDictReturnObject
::: lmetk.data.data_objects.data.ExtendedReturnObject

---

## Data object utilities

The module `lmetk.data.data_objects.utils` provides validation and conversion helpers used by data objects and transforms:

| Function | Description |
|----------|-------------|
| `convert_to_numpy(value, ensure_2d=True)` | Converts array-like data (NumPy, pandas DataFrame/Series, lists) to `np.ndarray`; optionally enforces 2D shape. |
| `check_length_consistency(values, keys, raise_on_error=True)` | Ensures all array-like items in `values` have the same length; optionally raises or warns. |
| `check_for_nans(values, keys, raise_on_error=True)` | Checks for NaN in numeric data (NumPy, pandas); optionally raises or warns. |
| `get_length(value)` | Returns the length of array-like `value` (e.g. first dimension) or `None` if not applicable. |

For full signatures and behavior, see the source in `lmetk/data/data_objects/utils.py`.

---

## Related documentation

- [Data objects & containers](dataobject.md) — Conceptual overview, hierarchy, and datasource → container flow.
- [Datasets](datasets.md) — Task-centric datasets that consume container output.
- [Datasets inner workings](datasets_inner_workings.md) — Data loading architecture (HydraConcat, sequencers, collation).
- [Evaluators](evaluators/index.md) — Evaluator API and configuration.
- [Transforms](transforms.md) — Transform system and base classes.
