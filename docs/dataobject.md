# Data Objects & Dataset Containers

The `lmetk.data.data_objects` module provides the fundamental data structures used to transport data throughout the PHM pipeline. Unlike standard Python dictionaries, these objects enforce **type safety**, provide **recursive dot-notation access**, and handle **metadata synchronization** automatically.

## 1. The Core Philosophy

In this framework, data is not passed around as loose tuples or plain dictionaries. It is encapsulated in **Containers**. This design ensures:

1.  **Strict Typing:** Fields can only hold `pandas.DataFrame`, `numpy.ndarray`, or `awkward.Array`. Lists of other objects are rejected at runtime.
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
* **Type Guarding:** Prevents assignment of invalid types. Fields must be numeric tensors or dataframes suitable for ML.

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

---

## 4. The Interplay: Datasource $\to$ Container

The transition from raw files to a structured container follows a strict ETL (Extract, Transform, Load) flow.

### The Flow Diagram

```text
Raw Files (HDF5/CSV) 
       ⬇
[ DataSourceLoader ] .load_data()
       ⬇ 
  (Internal Dict) 
  { 'features': {'train': [Array, ...]}, 'target': ... }
       ⬇
[ SplitDatasetContainer ] .get_data()
       ⬇ 
  (Validated Object)
  container.features.train[0] -> ak.Array
```

### Step-by-Step Interaction
Ingestion (AbstractDataSourceLoader): The loader reads raw bytes and performs the heavy lifting of reshaping flat signals into the framework's standard 3D jagged structure (Time, Freq, Channel). It builds a standard Python dictionary.

Encapsulation (SplitDatasetContainer): The loader instantiates the container with the internal dictionary.

```Python

# Inside DataSourceLoader.get_data()
return SplitDatasetContainer(**self.data_dict)
```
Recursive Parsing (__init__): The container instantly converts the nested dictionaries into nested BaseDataObjects, enabling dot-notation.

Validation (validate()): The container scans the structure. If the loader produced a ragged list where features has 50 units but target has 49, the container raises a ValueError immediately.
