# Examples

The examples below progress from the simplest case to fully custom data and models.
Each is self-contained and can be run as-is (assuming the package is installed and, where noted, the dataset is available).

---

## 1. Built-in datasource, built-in model

Train an LSTM on the PHME20 dataset for RUL prediction.
Everything uses defaults — the minimum viable call to `train()`.

```python
from lightning.pytorch.callbacks import EarlyStopping, RichModelSummary

from picid.interface import EntryInterface
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn

# --- transforms ---
scaler = DataTransform(
    transform_name="scaler_features",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "features", "fit_on": "train"},
)
scaler_target = DataTransform(
    transform_name="scaler_targets",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "rul", "fit_on": "train"},
)

# --- components ---
model           = LSTMConfig(n_layers=8)
task_definition = Prognostic(task_type="rul")
evaluators      = {s: RulEvaluatorConfig() for s in ["train", "val", "test"]}
loggers         = [CsvLogger(name="phme20_lstm")]
callbacks       = [EarlyStopping(monitor="val/loss"), RichModelSummary()]

# --- run ---
interface = EntryInterface()
interface.train(
    run_name="phme20_lstm_rul",
    model=model,
    task_definition=task_definition,
    datasource="phme20",              # built-in datasource name
    transforms=[scaler, scaler_target],
    evaluators=evaluators,
    loggers=loggers,
    callbacks=callbacks,
)
```

**Key points:**

- `datasource="phme20"` — reference a built-in datasource by name. Run `EntryInterface.get_available_datasources()` to list all available names.
- `transforms` are applied inside `train()` automatically.
- `overrides` is omitted — trainer defaults (CPU, 10 epochs) are used.

---

## 2. Pre-process first, then train

Separate the preprocessing step from training.
Useful when you want to inspect the preprocessed data or reuse it across multiple runs.

```python
from picid.interface import EntryInterface
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn

scaler = DataTransform(
    transform_name="scaler_features",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "features", "fit_on": "train"},
)
scaler_target = DataTransform(
    transform_name="scaler_targets",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "rul", "fit_on": "train"},
)
transforms = [scaler, scaler_target]

interface = EntryInterface()

# Step 1: load and preprocess
datasource          = interface.get_datasource("phme20")
processed_datasource = interface.process_datasource(datasource, transforms)

# Optional: inspect the data
print(processed_datasource.data_dict["train"]["features"].shape)

# Step 2: train — no transforms needed here, data is already processed
model           = LSTMConfig(n_layers=8)
task_definition = Prognostic(task_type="rul")

interface.train(
    run_name="phme20_preprocessed",
    model=model,
    task_definition=task_definition,
    datasource=processed_datasource,   # ProcessedDatasource — no transforms= needed
    evaluators={s: RulEvaluatorConfig() for s in ["train", "val", "test"]},
    loggers=[CsvLogger(name="phme20_preprocessed")],
)
```

---

## 3. Custom single-source datasource (NumPy)

Use your own NumPy array instead of a built-in dataset.

```python
import numpy as np
from picid.data.preprocessing import TimeSplitter
from picid.interface import EntryInterface, CustomSingleSourceLoader
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn

# --- your data: (n_samples, n_features + 1) ---
# last column is the RUL target
data = np.random.randn(2000, 11).astype(np.float32)

# --- splitter ---
splitter = TimeSplitter(
    train=0.6,
    val=0.2,
    test=None,     # remainder goes to test
    seq_len=16,
    pred_len=0,
    create_splits_for=["features", "timestamps", "rul"],
)

# --- datasource ---
datasource = CustomSingleSourceLoader.load_from_numpy(
    source=data,
    target_column=-1,    # last column is the target
    task_mode="rul",
    data_splitter=splitter,
)

# --- transforms ---
scaler = DataTransform(
    transform_name="scaler_features",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "features", "fit_on": "train"},
)
scaler_target = DataTransform(
    transform_name="scaler_targets",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "rul", "fit_on": "train"},
)

# --- preprocess and train ---
interface = EntryInterface()
processed = interface.process_datasource(datasource, [scaler, scaler_target])

interface.train(
    run_name="custom_numpy_rul",
    model=LSTMConfig(n_layers=4),
    task_definition=Prognostic(task_type="rul", seq_len=16),
    datasource=processed,
    evaluators={s: RulEvaluatorConfig() for s in ["train", "val", "test"]},
    loggers=[CsvLogger(name="custom_numpy")],
    overrides=["trainer.max_epochs=20"],
)
```

To use a pandas DataFrame instead, replace the loader call:

```python
import pandas as pd

df = pd.read_csv("my_data.csv")

datasource = CustomSingleSourceLoader.load_from_csv(
    source=df,
    target_column="rul",    # column name
    task_mode="rul",
    data_splitter=splitter,
)
```

---

## 4. Custom multi-source datasource

Use when you have multiple independent units (machines, subjects, trials).
Two splitting strategies are shown.

### 4a. Per-source splitting (each unit split independently)

```python
import numpy as np
from copy import deepcopy
from picid.data.preprocessing import TimeSplitter
from picid.interface import EntryInterface
from picid.interface.datasources import CustomMultiSourceLoader
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn

# --- three independent units ---
sources = {
    "unit_1": np.random.randn(800, 11).astype(np.float32),
    "unit_2": np.random.randn(600, 11).astype(np.float32),
    "unit_3": np.random.randn(1000, 11).astype(np.float32),
}

splitter = TimeSplitter(
    train=0.6, val=0.2, test=None,
    seq_len=16, pred_len=0,
    create_splits_for=["features", "timestamps", "rul"],
)
# each unit gets its own independent splitter instance
per_source_splitters = {k: deepcopy(splitter) for k in sources}

datasource = CustomMultiSourceLoader.load_from_primitive(
    sources=sources,
    target_column=-1,
    task_mode="rul",
    data_splitter=per_source_splitters,
)

scaler = DataTransform(
    transform_name="scaler_features",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "features", "fit_on": "train"},
)
scaler_target = DataTransform(
    transform_name="scaler_targets",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "rul", "fit_on": "train"},
)

interface = EntryInterface()
processed = interface.process_datasource(datasource, [scaler, scaler_target])

interface.train(
    run_name="multi_source_per_unit",
    model=LSTMConfig(n_layers=4),
    task_definition=Prognostic(task_type="rul", seq_len=16),
    datasource=processed,
    evaluators={s: RulEvaluatorConfig() for s in ["train", "val", "test"]},
    loggers=[CsvLogger(name="multi_source")],
    overrides=["trainer.max_epochs=20"],
)
```

### 4b. Global splitting by source assignment

Assign entire units to splits — unit 1 and 2 for training, unit 3 for testing:

```python
from picid.data.preprocessing import BySourceSplitter
from picid.interface.datasources import CustomMultiSourceLoader

splitter = BySourceSplitter(
    sources_train=["unit_1", "unit_2"],
    sources_val=[],           # empty val split
    sources_test=["unit_3"],
)

datasource = CustomMultiSourceLoader.load_from_primitive(
    sources=sources,
    target_column=-1,
    task_mode="rul",
    data_splitter=splitter,   # BySourceSplitter, not a dict
)
```

Then preprocess and train as in 4a.

---

## 5. Custom PyTorch model

Bring your own `nn.Module` for a RUL task.

```python
import torch.nn as nn
import numpy as np
from picid.data.preprocessing import TimeSplitter
from picid.interface import EntryInterface, CustomSingleSourceLoader
from picid.interface.model import CustomModelTrainer
from picid.interface.model.wrapper import ModelWrapper
from picid.interface.schemas.task_definition import Prognostic
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn


# --- 1. define your model ---
class MyMLP(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# --- 2. define the task ---
task = Prognostic(task_type="rul", seq_len=16)

# --- 3. instantiate and wrap the model ---
backbone = MyMLP(seq_len=task.seq_len, n_features=10, hidden=128)

# ModelWrapper adds a permute so the model receives (B, C, T) if needed.
# Here no reshape is needed — omit ModelWrapper or use identity function.
wrapped = ModelWrapper(model=backbone)

custom_model = CustomModelTrainer(task_type=task.task_type, model=wrapped)

# --- 4. prepare data ---
data = np.random.randn(2000, 11).astype(np.float32)

splitter = TimeSplitter(
    train=0.6, val=0.2, test=None,
    seq_len=task.seq_len, pred_len=0,
    create_splits_for=["features", "timestamps", "rul"],
)

datasource = CustomSingleSourceLoader.load_from_numpy(
    source=data, target_column=-1, task_mode="rul", data_splitter=splitter,
)

scaler = DataTransform(
    transform_name="scaler_features",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "features", "fit_on": "train"},
)
scaler_target = DataTransform(
    transform_name="scaler_targets",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "rul", "fit_on": "train"},
)

interface = EntryInterface()
processed = interface.process_datasource(datasource, [scaler, scaler_target])

# --- 5. train ---
interface.train(
    run_name="custom_mlp_rul",
    model=custom_model,        # CustomModelTrainer instance
    task_definition=task,      # required when using a custom model
    datasource=processed,
    evaluators={s: RulEvaluatorConfig() for s in ["train", "val", "test"]},
    loggers=[CsvLogger(name="custom_mlp")],
    overrides=["trainer.max_epochs=30"],
)
```

**Key points:**

- The `task_type` passed to `CustomModelTrainer` must match the `task_type` in `Prognostic`.
- `ModelWrapper` is optional; use it when your model expects a different tensor layout than what the dataloader provides.
- `task_definition` cannot be a raw dict when using `CustomModelTrainer`.
