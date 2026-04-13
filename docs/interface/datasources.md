# Custom Datasources

The interface provides two loader classes so you can bring your own data:

- **`CustomSingleSourceLoader`** — one contiguous dataset (a single machine, a single trial, etc.).
- **`CustomMultiSourceLoader`** — multiple independent datasets combined into one (e.g. multiple machines measured separately).

Both wrap your raw NumPy arrays or pandas DataFrames and expose the same interface that built-in datasources use.

```python
from picid.interface import CustomSingleSourceLoader
from picid.interface import CustomMultiSourceLoader
```

You can instantiate a SingleSourceLoader from a numpy array or a panda dataframe object. You can instantiate a MultiSource using the already create SingleSource or a collection of primitives.  

---

## Single-source loader

### From a NumPy array

```python
import numpy as np
from picid.data.preprocessing import TimeSplitter
from picid.interface import CustomSingleSourceLoader

# shape: (n_samples, n_features+1) — last column is the target
data = np.random.randn(1000, 11)

splitter = TimeSplitter(
    train=0.6,
    val=0.2,
    test=None,           # test gets the remainder
    seq_len=16,
    pred_len=0,
    create_splits_for=["features", "timestamps", "rul"],
)

loader = CustomSingleSourceLoader.load_from_numpy(
    source=data,
    target_column=-1,    # last column; negative indexing is supported
    task_mode="rul",     # key used for the target in the data dict
    data_splitter=splitter,
)
```

### From a pandas DataFrame

```python
import pandas as pd
from picid.interface import CustomSingleSourceLoader

df = pd.read_csv("my_data.csv")

loader = CustomSingleSourceLoader.load_from_csv(
    source=df,
    target_column="rul",   # column name or integer index
    task_mode="rul",
    data_splitter=splitter,
)
```

### From pre-split data

If you have already split your data into train/val/test sets, pass a dict:

```python
pre_split = {
    "train": train_array,   # np.ndarray or DataFrame
    "val":   val_array,
    "test":  test_array,
}

loader = CustomSingleSourceLoader.load_from_numpy(
    source=pre_split,
    target_column=-1,
    task_mode="rul",
    # no data_splitter needed — data is already split
)
```

The dict must contain exactly the keys `"train"`, `"val"`, and `"test"`.

---

### `load_from_numpy` and `load_from_csv` parameters

| Parameter | Type | Description |
|---|---|---|
| `source` | `np.ndarray \| dict[str, np.ndarray]` | Raw data or pre-split dict. |
| `target_column` | `int \| str` | Column that holds the prediction target. Negative integers are supported for NumPy. |
| `task_mode` | `str` | Key used for the target in the internal data dictionary (e.g. `"rul"`, `"target"`, `"forecasting"`). Must match the `task_type` of your `BaseTaskDefinition`. |
| `data_splitter` | `Callable \| None` | Splitting strategy. Required unless data is already a pre-split dict. |
| `data_name` | `str` | Human-readable label used in log messages. |

---

## Multi-source loader

`CustomMultiSourceLoader` wraps several independent sources, identified by string keys.
It is the right choice when each "unit" (machine, subject, trial) was recorded independently and should not be concatenated before splitting.

### From a dict of arrays

```python
from copy import deepcopy
from picid.interface.datasources import CustomMultiSourceLoader
from picid.data.preprocessing import TimeSplitter

arrays = {
    "unit_1": np.random.randn(800, 11),
    "unit_2": np.random.randn(600, 11),
    "unit_3": np.random.randn(1200, 11),
}

# Option A — per-source splitting (each unit is split independently)
splitter = TimeSplitter(train=0.6, val=0.2, test=None,
                        seq_len=16, pred_len=0,
                        create_splits_for=["features", "timestamps", "rul"])

per_source_splitters = {k: deepcopy(splitter) for k in arrays}

loader = CustomMultiSourceLoader.load_from_primitive(
    sources=arrays,
    target_column=-1,
    task_mode="rul",
    data_splitter=per_source_splitters,   # dict[str, splitter]
)
```

### Global splitting by source assignment

When units should be assigned wholesale to one split (train on units A & B, test on unit C):

```python
from picid.data.preprocessing import BySourceSplitter

splitter = BySourceSplitter(
    sources_train=["unit_1", "unit_2"],
    sources_val=[],
    sources_test=["unit_3"],
)

loader = CustomMultiSourceLoader.load_from_primitive(
    sources=arrays,
    target_column=-1,
    task_mode="rul",
    data_splitter=splitter,   # BySourceSplitter — not a dict
)
```

When a `BySourceSplitter` is used, individual loaders must **not** carry their own `data_splitter`.
The two approaches are mutually exclusive.

---

### `load_from_primitive` parameters

| Parameter | Type | Description |
|---|---|---|
| `sources` | `dict[str, np.ndarray \| DataFrame]` | Mapping from source name to raw data. |
| `target_column` | `int \| str` | Target column, shared across all sources. |
| `task_mode` | `str` | Task key (same for all sources). |
| `data_splitter` | `dict[str, splitter] \| BySourceSplitter \| None` | Either a per-source dict of splitters, or a single `BySourceSplitter`. |
| `data_name` | `str` | Human-readable label. |

---

## Splitting strategies

### `TimeSplitter`

Splits a single contiguous time-series by ratio.

```python
from picid.data.preprocessing import TimeSplitter

splitter = TimeSplitter(
    train=0.6,    # fraction of samples for training
    val=0.2,      # fraction for validation
    test=None,    # test = 1 - train - val
    seq_len=16,   # must match task definition
    pred_len=0,   # must match task definition
    create_splits_for=["features", "timestamps", "rul"],  # keys to split
)
```

Use with `CustomSingleSourceLoader` or as per-source splitters in `CustomMultiSourceLoader`.

### `BySourceSplitter`

Assigns entire sources to splits — no time-based splitting within a source.

```python
from picid.data.preprocessing import BySourceSplitter

splitter = BySourceSplitter(
    sources_train=["unit_1", "unit_2"],
    sources_val=["unit_3"],
    sources_test=["unit_4"],
)
```

Use only with `CustomMultiSourceLoader`.

---

## Processing and passing data to `train()`

After creating a loader you pass it to `process_datasource()`, which runs the full preprocessing pipeline (load → split → apply transforms) and returns a `ProcessedDatasource`.

```python
from picid.interface import EntryInterface
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn

interface = EntryInterface()

scaler = DataTransform(
    transform_name="scaler_features",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "features", "fit_on": "train"},
)

processed = interface.process_datasource(loader, transforms=[scaler])

# processed is a ProcessedDatasource — pass it directly to train()
interface.train(
    ...,
    datasource=processed,
    # no transforms= needed here — data is already transformed
)
```

### Why pre-process separately?

- **Re-use:** Run `process_datasource()` once and pass `processed` to multiple `train()` calls with different models.
- **Inspection:** Check `processed.data_dict` to verify the shapes and values before training.
- **Caching:** The preprocessor uses `joblib.Memory`, so repeated calls with the same data and transforms are served from disk.

### `ProcessedDatasource` attributes

| Attribute | Description |
|---|---|
| `data_dict` | Nested dict `{split → {key → array}}`, e.g. `data_dict["train"]["features"]`. |
| `meta_data_dict` | Metadata returned by the datasource (unit ids, normalization stats, etc.). |
| `task_mode` | The `task_mode` string from the loader. |
| All other attributes | Forwarded transparently to the underlying `DatasetContainer`. |
