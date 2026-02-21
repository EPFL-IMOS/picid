# Data Objects & Dataset Containers

The `picid.data.data_objects` module provides the fundamental data structures used to transport data throughout the PHM pipeline. Unlike standard Python dictionaries, these objects enforce **type safety**, provide **recursive dot-notation access**, and handle **metadata synchronization** automatically.

## 1. The Core Philosophy

In this framework, data is not passed around as loose tuples or plain dictionaries. It is encapsulated in **Containers**. This design ensures:

1.  **Strict Typing:** Fields can only hold `pandas.DataFrame`, `pandas.Series`, `numpy.ndarray`, or `awkward.Array`—or lists of these (e.g. one list per unit). Other types raise `TypeError` at assignment.
2.  **Structure Validation:** The container guarantees that if you have 100 engines in your training set, you also have 100 target labels.
3.  **Metadata Persistence:** Column names and unit identifiers travel attached to the data, preventing "magic number" indexing bugs.

---

## 2. Visualizing the Data Structure

The `SplitDatasetContainer` is versatile. It stores data in a column-oriented format but allows you to pivot it depending on your needs.

### A. The Container View (Standard Access)
*How the data looks when you access it directly via dot-notation.*

```python
container.features.train
# └── List[ak.Array] (Length: 100 Units)
#     ├── Unit 0: Array(shape=(60, 1024, 1))
#     ├── Unit 1: Array(shape=(45, 1024, 1))
#     └── ...

container.target.train
# └── List[ak.Array] (Length: 100 Units)
#     ├── Unit 0: Array(shape=(60, 1, 1))
#     ├── Unit 1: Array(shape=(45, 1, 1))
#     └── ...
```
### B. The Machine View (to_split_dict)
Used by PyTorch/Hydra. Organized by Split $\to$ Field $\to$ List of Units
```
# Output of container.to_split_dict()
{
    "train": {
        "features": [Array_Unit0, Array_Unit1, ...],  # All feature arrays
        "target":   [Array_Unit0, Array_Unit1, ...]   # All target arrays
    },
    "test": {
        "features": [Array_Unit0, ...],
        "target":   [Array_Unit0, ...]
    }
}
```

### C. The Analyst View (group_by_split)
Used for plotting/debugging. Organized by Split $\to$ Unit Object.

```
# Output of container.group_by_split()
{
    "train": [
        # Unit 0 Object
        BaseDataObjectWithMetadata(
            features=Array_Unit0,
            target=Array_Unit0,
            metadata={...}
        ),
        # Unit 1 Object
        BaseDataObjectWithMetadata(
            features=Array_Unit1,
            target=Array_Unit1,
            metadata={...}
        )
    ]
}
```

## 3. Class Hierarchy

The data objects follow a strict inheritance ladder, each adding specific capabilities.

### `BaseDataObject` (The Enforcer)
*Replaces standard Python `dict`.*
* **Recursive Construction:** Nested dictionaries `{'a': {'b': 1}}` are automatically converted into nested objects, enabling `obj.a.b` access.
* **Type Guarding:** Prevents assignment of invalid types. Fields must be one of: `pandas.DataFrame`, `pandas.Series`, `numpy.ndarray`, or `awkward.Array` (or lists of these). Other types raise `TypeError` at assignment.

### `BaseDataObjectWithMetadata` (The Archivist)
*Inherits from `BaseDataObject`.*
* **Auto-Column Mapping:** Automatically extracts column names from assigned `pandas.DataFrame`s and stores them in a hidden `_metadata` dictionary.
* **Consistency:** Keeps bookkeeping separate from mathematical operations.

### `DatasetContainer` (The Normalizer)
*Inherits from `BaseDataObjectWithMetadata`.*
* **Batch-of-1 Handling:** Automatically detects single-unit loads and wraps them in a list, ensuring downstream pipelines always iterate over lists.
* **Multi-Unit Flag:** Includes the `is_multi_unit` flag (defaulting to `True` for industrial fleet data).

### `SplitDatasetContainer` (The Standard Output)
*Inherits from `DatasetContainer`.*
* **Cross-Split Validation:** Its `validate()` method ensures semantic consistency (e.g., `features.train` and `target.train` must have the same number of units).

### SliceInfo (optional container-level metadata)

**What it is:** `SliceInfo` is an optional dataclass attached to the container (`container.slice_info`) that describes *which slice of the dataset* the container represents (e.g. which split, which unit IDs, which cycles).

**Fields (all optional):** `split` (e.g. `"train"`, `"val"`, `"test"`), `unit_ids`, `cycle_ids`, `bounds`, `index_map`.

**How it is created:** It is **not** set automatically. A datasource or preprocessing step can set it when building the container by passing `slice_info=SliceInfo(...)` into `SplitDatasetContainer(...)`. If not set, `container.slice_info` is `None`. See `picid.data.data_objects.slice_info.SliceInfo`.

**Transforms:** Transforms can receive this information in their `metadata` dict (under the key `"slice_info"`, as a serialized dict) by opting in in config: see the transform metadata key `include_slice_info_in_metadata` in the [Transforms guide](../transforms/guide.md).

---

## 4. The Interplay: Datasource $\to$ Container

The transition from raw files to a structured container follows a strict ETL (Extract, Transform, Load) flow.

### The Flow Diagram

```text
Raw Files (HDF5/CSV, etc.)
       ⬇
[ DataSourceLoader ].load_data()     ← Populates internal state (e.g. self.data_dict)
       ⬇
  (Internal dict)
  { 'features': {'train': [Array, ...], 'val': ..., 'test': ...}, 'target': ... }
       ⬇
[ DataSourceLoader ].get_data()      ← Returns the container to the caller
       ⬇
  return SplitDatasetContainer(**self.data_dict)
       ⬇
  (Validated object)
  container.features.train[0]  →  ak.Array  (e.g. one unit’s feature array)
```

### Step-by-step interaction

1. **Ingestion (`load_data()`):** The loader reads raw files and reshapes data into the framework’s standard structure (e.g. per-split, per-unit lists of arrays). It fills an internal dictionary (e.g. `self.data_dict`) but does not yet return a container.
2. **Encapsulation (`get_data()`):** The loader instantiates `SplitDatasetContainer(**self.data_dict)` and returns it. Callers (e.g. the pipeline or a datamodule) typically call `load_data()` once, then `get_data()` to obtain the container.
3. **Recursive parsing (`__init__`):** The container converts nested dicts into nested `BaseDataObject` instances, so you can use dot-notation (e.g. `container.features.train`).
4. **Validation (`validate()`):** The container checks that, for each split, all fields (e.g. `features`, `target`) have the same number of units. If, for example, `features.train` has 50 units but `target.train` has 49, `validate()` raises `ValueError`.

---

## 5. Module location and API

All types above live in the package **`picid.data.data_objects`**:

- **`base.py`** — `BaseDataObject`, `BaseDataObjectWithMetadata`, `ToNumpyMixin`, `ToDataFrameMixin`
- **`data.py`** — `NamedTransformInput`, `DatasetContainer`, `SplitDatasetContainer`, and the generic return types (`ReturnObject`, `SimpleReturnObject`, etc.)
- **`utils.py`** — `convert_to_numpy`, `check_length_consistency`, `check_for_nans`, `get_length`

For method signatures and full API details, see the [API Reference](api.md).
