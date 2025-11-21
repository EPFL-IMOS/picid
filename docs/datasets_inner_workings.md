# Datasets and Data Loading Architecture

In this project, we adopt the notation of a **Dataset** from PyTorch. A PyTorch `Dataset` is an abstract class representing a collection of data samples. It defines *how* data is accessed and transformed, enabling efficient iteration for training models.

> **Concept:** In this framework, a **Datasource** is data-centric (raw files, IO), whereas a **Dataset** is task-centric (formatting data for specific model inputs).

We provide a modular hierarchy of datasets designed to handle complex, multi-unit time-series data (e.g., industrial fleets, sensor logs).

---

## 1. HydraConcatDataset ( The Root Orchestrator )

`HydraConcatDataset` acts as the root container for the entire data loading pipeline. It wraps PyTorch's `ConcatDataset` but handles the complex initialization logic required for multi-unit data.

### Role
Its primary job is to take a dictionary of **Lists of Arrays** (where each item in the list corresponds to a specific unit/engine) and instantiate a specific child dataset for each unit.

### Data Hierarchy
The data loading pipeline follows a strict hierarchy:

1.  **HydraConcatDataset (Root)**:
    -   **Input**: A dictionary where keys are modalities (e.g., `'features'`) and values are **Lists** of arrays (e.g., 100 engines).
    -   **Action**: Iterates over the list of units and instantiates a Child Dataset for each one.
    -   **Output**: A single, indexable view of all windows across all units.

2.  **Child Dataset (e.g., ContextBatchDataset)**:
    -   **Input**: A single unit's arrays (e.g., Engine 1's features & targets).
    -   **Action**: Synchronizes different modalities. It ensures that if we grab window `k`, we get the 'features' window and the 'target' window for the exact same time steps.
    -   **Mechanism**: Instantiates two `SlidingWindowBatchDataset`s internally.

3.  **SlidingWindowBatchDataset (Leaf)**:
    -   **Input**: A single array for a single modality.
    -   **Action**: Handles the mechanics of slicing (seq_len, stride).
    -   **Mechanism**: Wraps the `Sequencer`.

4.  **Sequencer (e.g., RaggedArraySequencer)**:
    -   **Input**: The raw array (Awkward or Numpy).
    -   **Action**: Pre-computes valid start indices for every possible window.

---

## 2. ContextBatchDataset

`ContextBatchDataset` is the standard "Child Dataset" designed for sequence tasks where both **target** (output) and **context** (input) features are needed.

It wraps two `SlidingWindowBatchDataset` instances—one for the target and one for the context—using shared time parameters (sequence length, prediction length, stride).

### Batch Contents

Each batch from `ContextBatchDataset` is returned as an `AttributeDict` with the following structure:

```python
{
    'batch_idx': [indices],
    'target': {
        # The target sequence (e.g., RUL values)
        'rul_seq_x': <target input sequence>,
        'rul_seq_y': <target output sequence (prediction horizon)>
    },
    'context': {
        # The context sequence (e.g., Sensor readings)
        'features_seq_x': <context input sequence>,
        'features_seq_y': <context output sequence>
    }
}
```

## 3. RULContextBatchDataset ( Specialized Task )

`RULContextBatchDataset` is a specialized version of `ContextBatchDataset` designed specifically for **Remaining Useful Life (RUL) prediction**.

-   **Inheritance**: It inherits from `ContextBatchDataset`, utilizing the same underlying mechanism for handling target and context features.
-   **Sliding Window Processing**: It moves a sliding window of a defined length (`seq_len`) over the time-series to generate samples.
-   **RUL-Specific Target**: The core feature is that the model is trained to predict the RUL at the **final time step** of the input sequence. This is achieved by internally forcing the prediction length (`pred_len`) to 0, which aligns the target window perfectly with the input window.

### Batch Contents

Since it inherits from `ContextBatchDataset`, each batch is returned as an `AttributeDict` with a standardized structure. For RUL, the operational **features** are typically treated as the `context` and the **RUL values** as the `target`.

The output batch structure is optimized for RUL regression:

```python
{
    'batch_idx': [indices],
    # The RUL value at the very end of the window. Shape: (Batch, 1)
    'rul': target['rul_seq_x'][:, -1, :],   
    
    # The history window of sensor readings. Shape: (Batch, seq_len, Channels)
    'features': context['features_seq_x']   
}
```
### The Role of `pred_len=0`

Setting **`pred_len=0`** is the configuration that defines this specific task type.

-   **Forecasting (`pred_len > 0`)**: This setup implies "Predict values `pred_len` steps *after* the input ends." If your input sequence ends at time `t`, setting `pred_len=5` means your target is the value at time `t+5`.
-   **RUL Prediction (`pred_len = 0`)**: This setup implies "The target value exists at the *exact same time step* where the input ends." Since RUL is a property of the current state (time `t`), we do not want to look into the future; we want the label corresponding to the last observed moment.

---


## Ragged vs Dense: A Structural Distinction

Ragged vs Dense is about **regularity of lengths along the time axis** — not about how many units you have, and not about whether the data is single-unit or multi-unit.

### Dense arrays

- All sequences have the **same length** along the time dimension.
- They can represent:
  - a single continuous sequence: `(T, F)`
  - or many units that all share the same sequence length: `(Units, T, F)`
- These arrays are **regular** (non-jagged). They can be stored in NumPy or as a regular Awkward array.
- Internally, our drivers may *mark* the time axis as ragged for sequencing, but this is purely an implementation detail—the data itself is still dense/regular.

### Ragged arrays

- The number of dimensions is the same, but lengths along the time axis **vary per unit**.
- Typical shape: `(Units, varT, F)` where each unit has its own sequence length `T_i`.
- This requires Awkward’s **jagged** representation.
- The sequencer treats each unit **independently**, enforcing:
  - Windows cannot cross unit boundaries,
  - Padding only occurs when explicitly enabled,
  - Each unit must individually satisfy the required length:  
    `seq_len + pred_offset + pred_len`.

### Summary

- **Dense** = every sequence has the same length along the time axis.
- **Ragged** = sequences have different lengths.
- **Multi-unit or single-unit is a separate concept** — both can be dense or ragged.

This clear separation is essential because sequencing logic depends on the regularity of the time dimension, not the number of units.


---


## 4. The Sequencing Engine

At the lowest level, `LMETK` uses specialized sequencers to generate valid window indices from raw data arrays without duplicating memory.

### AkwardJaggedAutoregressiveSequencer ( The Engine )

This class is the computational core. It calculates valid start indices for sliding windows and is robust against irregular (jagged) data structures.

**Key Features:**
* **Jagged Support:** Handles multiple units (e.g., engines) with different temporal lengths seamlessly. It ensures windows do not cross unit boundaries.
* **Strict Windowing (Default):** Drops units that are too short to form a full window of history.
* **Padding/Warmup:** Supports starting predictions earlier by zero-padding the history (Left Padding).
* **Vectorized Indexing:** Pre-computes all valid indices into a NumPy array for $\mathcal{O}(1)$ access during training.



**Example Scenario:**
Data: 3 Units with lengths `[5, 3, 6]`. Configuration: `seq_len=3`, `pred_len=2`. Required Length: 5.

* **Unit 0 (Len 5):** $5 \ge 5$. **Valid.** 1 Window generated.
* **Unit 1 (Len 3):** $3 < 5$. **Invalid.** 0 Windows generated (Dropped).
* **Unit 2 (Len 6):** $6 \ge 5$. **Valid.** 2 Windows generated.



### The Drivers: Dense vs. Ragged

To make the engine easy to use and robust against different input formats, we wrap it in two driver classes:

#### 1. DenseArraySequencer
Designed for single time-series data (e.g., one long continuous sensor recording).
* **Input:** Numpy Array or PyTorch Tensor of shape `(Time, Feat)`.
* **Logic:** Converts the input to an Awkward Array and forces **`axis=0` (Time)** to be treated as the variable dimension.
* **Result:** The engine sees "1 Unit" of length $T$.

#### 2. RaggedArraySequencer
Designed for multi-unit data (e.g., a batch of different engines).
* **Input:** Awkward Array of shape `(Batch, var_Time, Feat)` or a regular `(Batch, Time, Feat)`.
* **Logic:** Detects the ragged dimension. If the input is regular (`(B, T, F)`), it explicitly marks the time dimension (**Axis 1**) as ragged to preserve the "Multi-Unit" structure.
* **Result:** The engine iterates over $B$ units, calculating windows for each unit independently.






# Deep Dive: The `AkwardJaggedAutoregressiveSequencer`

The `AkwardJaggedAutoregressiveSequencer` is the computational engine behind data loading in LMETK. Unlike standard NumPy striding tricks which assume a single contiguous block of memory, this sequencer is designed to handle **Jagged (Multi-Unit)** time-series data efficiently.

It solves the problem of generating valid sliding windows (Input $X$, Target $y$) from a dataset containing multiple independent time series (e.g., different engines, batteries, or sensors) without padding the raw data or concatenating unrelated units.

---

## 1. Core Responsibilities

The sequencer performs three critical tasks during initialization:

1.  **Structure Detection**: It analyzes the input `Awkward Array` to determine if it is **Dense** (single time series) or **Jagged** (multiple units).
2.  **Index Pre-computation**: It calculates a master list of valid `(Unit_Index, Start_Time)` tuples. This allows for $\mathcal{O}(1)$ random access during training via `__getitem__`.
3.  **Window Validation**: It enforces rules (Sequence Length, Prediction Horizon, Stride) to ensure every generated window contains valid data (or valid padding).

---

## 2. How Index Building Works

The most complex part of the sequencer is `_build_indices`. This method determines where a window can validly start.

### The Logic of "Valid Starts"

For a window to be valid, the data segment must be long enough to contain:
1.  The **History** (`seq_len`)
2.  The **Gap** (`pred_offset`, usually 0)
3.  The **Prediction** (`pred_len`)

$$\text{Required Length} = \text{seq\_len} + \text{pred\_offset} + \text{pred\_len}$$

### A. Strict Windowing (Default)

In strict mode (`padding_left_flag=False`), the sequencer drops any unit shorter than the required length. A window cannot start before time index 0.

* **Start Constraint:** $\text{start\_index} \ge 0$
* **End Constraint:** $\text{start\_index} + \text{Required Length} \le \text{Unit Length}$

> 

**Example:**
* Unit Length: 5 `[A, B, C, D, E]`
* Seq: 3, Pred: 2 $\rightarrow$ Required: 5.
* **Valid Start:** Index 0 only. (`[A, B, C]` $\rightarrow$ `[D, E]`)

### B. Padding / Warmup Mode

In padding mode (`padding_left_flag=True`), we relax the start constraint. We allow the input window to partially "overhang" the beginning of the data (negative indices), provided the **Target** exists within the real data.

This is controlled by `warmup_steps`: the index where the *first prediction* should occur.

* **Start Constraint:** $\text{start\_index} = \text{warmup\_steps} - \text{seq\_len} - \text{pred\_offset}$
* **Effect:** If `start_index` is negative (e.g., -2), the system will zero-pad the first 2 steps of the input.

> 

---

## 3. Data Retrieval Strategy (`_get_window_for_index`)

When the DataLoader requests a batch of indices, the sequencer performs a vectorized fetch. This method is optimized to avoid Python loops over batch items.

### Step 1: Coordinate Expansion
The sequencer retrieves the pre-computed tuples.
* Input: Batch of indices `[0, 5, 10]`
* Output: A tuple of arrays `(Unit_Indices, Time_Start_Indices)`

### Step 2: Vectorized Slicing
It constructs relative range arrays to slice the data.

* **X Range:** `[0, 1, ..., seq_len-1]` + `Time_Start_Indices`
* **Y Range:** `[offset, ..., offset + pred_len]` + `Time_Start_Indices` + `seq_len`

### Step 3: The Padding Trick (Masking)
If padding is enabled, `x_range` might contain negative numbers (e.g., `[-2, -1, 0, 1]`). Standard array slicing would interpret `-1` as the *last* element (wrapping), which is wrong for time series.

To solve this efficiently:
1.  **Clamp:** We force all negative indices to 0 (`safe_idx = max(idx, 0)`).
2.  **Fetch:** We grab data. The "negative" slots now contain data from index 0 (incorrect).
3.  **Mask:** We calculate a boolean mask (`original_idx >= 0`).
4.  **Apply:** We multiply the fetched tensor by the mask. The incorrect data becomes **0.0**.

This ensures high-performance, vectorized zero-padding without manual memory copying.

---

## 4. Summary of Modes

| Feature | Dense Mode | Ragged Mode |
| :--- | :--- | :--- |
| **Input Shape** | `(Time, Feat)` | `(Units, Time, Feat)` |
| **Interpretation** | Single long history | $N$ independent engines/cycles |
| **Iteration** | Slides over Time dim | Iterates Units, then slides over Time |
| **Padding** | Valid if `Total > Target` | Valid if `Unit_Len > Target` |

This architecture allows `LMETK` to treat massive, multi-unit industrial datasets as easily as a single small NumPy array, abstracting away the complexity of ragged indexing from the model training loop.


# The Drivers: DenseArraySequencer and RaggedArraySequencer

While the `AkwardJaggedAutoregressiveSequencer` is the computational "Engine," it requires inputs in a specific format (Awkward Arrays with specific ragged dimensions). To make the library user-friendly, we wrap this engine with two **Driver Classes**.

These drivers act as **Adapters**. They handle:
1.  **Input Validation:** Ensuring data is valid (Numpy, Torch, or Awkward).
2.  **Type Casting:** Converting everything to `float32`.
3.  **Structure Enforcement:** Explicitly telling the engine which dimension represents "Time" by marking it as ragged.

---

## 1. DenseArraySequencer

This driver is designed for **Single Time-Series** data (Dense). This is common in datasets where you have one long continuous recording (e.g., a single sensor log) rather than multiple distinct episodes.

### Inner Workings
1.  **Input Normalization**: Accepts `np.ndarray` or `torch.Tensor`. It immediately converts them into an `ak.Array`.
2.  **Axis 0 Regularization**:
    The engine expects a "Ragged" dimension to define the sequence length. For a dense array of shape `(Time, Features)`, the time dimension is **Axis 0**.
    The driver calls:
    ```python
    array = ak.from_regular(array, axis=0)
    ```
    This transforms the type from `N * M * type` to `var * M * type`. Even though the length is fixed, marking it as `var` tells the engine: **"Iterate over Axis 0 as the time dimension."**

### Interpretation
* **Input:** `(1000, 14)` (1000 time steps, 14 features).
* **Engine View:** The engine treats this as **1 Unit** with a duration of **1000**.
* **Result:** Valid windows are sliding slices of this single unit.

---

## 2. RaggedArraySequencer

This driver is designed for **Multi-Unit / Multi-Cycle** data. This is the standard for Predictive Maintenance (PdM) where you have data from multiple engines (e.g., Engine 1 runs for 50 cycles, Engine 2 for 200 cycles).

### Inner Workings
1.  **Input Validation**: Expects an `ak.Array`.
2.  **Ragged Dimension Detection**: It calls `find_singular_ragged_dim`.
    * If the array is already jagged (e.g., `(100, var, 14)`), the engine automatically detects the time dimension.
3.  **Regular Fallback (The "Regular as Jagged" Case)**:
    Sometimes, you might pass a "Multi-Unit" dataset where every unit happens to have the exact same length (e.g., `(100, 50, 14)`). Technically, this is a regular array.
    * The driver detects `ragged_dim is None`.
    * It effectively forces the array to be treated as jagged to preserve the hierarchical structure used by the engine.

### Interpretation
* **Input:** `(100, var, 14)` (100 Units, variable time, 14 features).
* **Engine View:** The engine iterates over **100 Units**.
* **Result:**
    * It calculates windows for Unit 1.
    * It resets.
    * It calculates windows for Unit 2.
    * It **never** creates a window that bleeds from Unit 1 into Unit 2.

---

## Summary Comparison

| Feature | DenseArraySequencer | RaggedArraySequencer |
| :--- | :--- | :--- |
| **Input Type** | `numpy.ndarray` / `torch.Tensor` | `awkward.Array` |
| **Typical Shape** | `(Time, Features)` | `(Batch/Units, Time, Features)` |
| **Topology** | Single Continuous Series | Multiple Independent Series |
| **Ragged Axis** | Forced on Axis 0 | Detected (or Forced) |
| **Use Case** | Forecasting on a single signal | RUL / Classification on fleet data |


# The Leaf Dataset: SlidingWindowBatchDataset

The `SlidingWindowBatchDataset` is the fundamental building block of the LMETK data loading hierarchy. It is responsible for wrapping raw data arrays into Sequencers and ensuring that different data modalities (like features and targets) remain perfectly synchronized in time.

It acts as a **Manager of Sequencers**. Instead of handling just one array, it handles a dictionary of arrays, creating a specific `Sequencer` (Driver) for each one.

---

## 1. Initialization Logic: The Auto-Detection Factory

When you initialize this dataset, it iterates through the provided `data_dict`. For every key-value pair, it automatically decides which Driver to use based on the data type.

### Type Dispatching
* **Numpy / PyTorch Tensor** $\rightarrow$ `DenseArraySequencer`
* **Awkward Array** $\rightarrow$ `RaggedArraySequencer`

This allows you to mix data types within a single dataset. For example, you might have **Dense** sensor readings (regular sampling) but **Ragged** event logs (irregular length), and this class will handle both side-by-side.

### Synchronization Safety Check
A critical responsibility of this class is ensuring **Temporal Alignment**.
If you provide `features` and `targets`, they must yield the **exact same number of windows**.

```python
# Internal Logic Check
if self._length is None:
    self._length = len(driver)
elif len(driver) != self._length:
    # CRITICAL ERROR: Mismatch. 
    # Example: Features has 100 steps, Targets has 99 steps.
    # This raises ValueError to prevent training on misaligned labels.
    raise ValueError(...)
```

## 2. Data Retrieval (`__getitem__`)

When the PyTorch `DataLoader` requests a sample at a specific index (e.g., `idx=5`), the `SlidingWindowBatchDataset` acts as a broadcaster. It does not slice the data itself; instead, it delegates the request to every internal sequencer it manages.



### The Retrieval Flow
1.  **Input:** The dataset receives an index (or a batch of indices).
2.  **Broadcast:** It loops through every registered key (e.g., `"features"`, `"targets"`) and calls `sequences_batch(idx)` on the corresponding driver.
3.  **Aggregation:** It collects the results into a single flat dictionary.

### The Suffix System
Since every sequencer returns a tuple of `(Input, Prediction)`, we need a way to store them in a single dictionary without naming collisions. The dataset applies **Suffixes** (default: `_seq_x` and `_seq_y`).

**Example:**
If you initialize the dataset with:
```python
data_dict = {
    "sensor_readings": <Array>,
    "rul_label": <Array>
}
```
The __getitem__ output will look like this:
```python
{
    # From 'sensor_readings' sequencer
    "sensor_readings_seq_x": Tensor(...),  # The historical window (Context)
    "sensor_readings_seq_y": Tensor(...),  # The future window (if pred_len > 0)

    # From 'rul_label' sequencer
    "rul_label_seq_x": Tensor(...),        # The historical RUL trajectory
    "rul_label_seq_y": Tensor(...)         # The future RUL trajectory
}
```

This flat structure makes it incredibly easy to feed into models that expect specific named inputs (e.g., HuggingFace-style forward passes).

## 3. Custom Collation (`collate_key_value_batch`)

The project utilizes a specialized collation function, `collate_key_value_batch`, designed to handle batches where items are dictionaries containing Tensors or Tuples of Tensors.

Unlike standard PyTorch collation which uses `torch.stack`, this function uses **`torch.cat`**. This is essential because our Sequencers produce pre-batched tensors (dimension 0 is already a batch dimension), so we must concatenate them rather than stack them to avoid creating an extra dimension.

### Logic Flow
The function iterates through the batch of dictionaries and separates values into two categories:
1.  **Tensors:** Collected into a simple list.
2.  **Tuples:** Collected into a list for unzipping (useful if a dataset returns `(input, target)` tuples directly).

### Aggregation Strategy
-   **Tensors:** Concatenated along dimension 0 (`torch.cat`).
-   **Tuples:** Transposed (unzipped) so that all first elements are grouped and all second elements are grouped, then each group is concatenated.

### Limitations
-   **Flat Structure Only:** This implementation **does not** recursively traverse nested dictionaries. It expects the input dictionary values to be Tensors or Tuples.
-   **Compatibility:** While it works perfectly for `SlidingWindowBatchDataset` (which outputs a flat dict), it **must be updated** to support recursion if used with `ContextBatchDataset` (which outputs nested `target`/`context` dictionaries).



```python
# Example Behavior
batch = [
    {"a": Tensor([1, 2])}, 
    {"a": Tensor([3, 4])}
]
# Output:
# {"a": Tensor([1, 2, 3, 4])}
```


# 4. The Orchestrator: ContextBatchDataset

While `SlidingWindowBatchDataset` handles the low-level slicing of specific arrays, `ContextBatchDataset` is the **Task-Level Orchestrator**. It organizes data into the semantic groups required by machine learning models: **Context** (Input) and **Target** (Output).

It acts as a wrapper that manages **two** internal `SlidingWindowBatchDataset` instances simultaneously, ensuring they remain perfectly synchronized.

## 1. Initialization and Task Mapping

The dataset is initialized with a raw `data_dict` containing arrays. Based on the `task_type` argument, it automatically selects which arrays serve as inputs and which serve as targets.

| Task Type | Context Key | Target Key | Use Case |
| :--- | :--- | :--- | :--- |
| `rul`, `ahrul`, `soc` | `features` | `rul` | Remaining Useful Life Prediction |
| `forecasting` | `features` | `target` | Time Series Forecasting |
| `fault_classification` | `features` | `fault_classification` | Fault Diagnosis |

### Synchronization Safety Check
A critical feature of this class is **Length Synchronization**.
During initialization, it asserts that the `context_dataset` (Inputs) and `target_dataset` (Outputs) produce the **exact same number of sliding windows**.

```python
assert len(self.target_dataset) == len(self.context_dataset)
```

If the features generate 100 windows but the targets generate 99 (e.g., due to array length mismatch or different sampling frequencies), initialization fails immediately. This prevents training on misaligned labels.

## 2. Nested Data Retrieval

The `__getitem__` method of `ContextBatchDataset` is the retrieval hub. When a `DataLoader` requests an index (or a list of indices), this method performs a synchronized fetch from both the `context_dataset` and `target_dataset`.

It structures the result into a hierarchical `AttributeDict`, which allows downstream models to access data using dot notation (e.g., `batch.context.features`).

### The Retrieval Flow
1.  **Input:** Accepts a `batch_idx` (integer or list of integers).
2.  **Delegation:** It passes this index to both internal datasets:
    * `self.target_dataset[batch_idx]`
    * `self.context_dataset[batch_idx]`
3.  **Aggregation:** It wraps the results into semantic dictionaries.

### Output Structure
The returned object is a nested dictionary structure containing the sliced tensors.

```python
{
    # The indices requested for this batch
    "batch_idx": [0, 1, ...], 
    
    "context": {
        # Data fetched from the 'context' dataset (e.g., Sensor readings)
        # Shape: (Batch, Seq_Len, Features)
        "features_seq_x": Tensor(...), # Input History
        "features_seq_y": Tensor(...)  # Input Future (usually unused/empty)
    },
    
    "target": {
        # Data fetched from the 'target' dataset (e.g., RUL or Forecast Targets)
        # Shape: (Batch, Seq_Len, 1) or (Batch, Pred_Len, 1)
        "rul_seq_x": Tensor(...),      # Target History
        "rul_seq_y": Tensor(...)       # Target Future (Prediction Horizon)
    }
}
```

This separation ensures that the model receives a clean, structured input where inputs **(context)** are clearly distinguished from supervision signals **(target)**, facilitating modular training loops.

# Conclusion

The LMETK data loading architecture is built on a philosophy of **separation of concerns**. By breaking down the complex problem of multi-unit, irregular time-series sampling into distinct layers, we achieve high flexibility and robustness without sacrificing performance.

### The Hierarchy Recap

This modular design ensures that no single class becomes a "God Object." Instead, each layer has a specific, testable responsibility:

1.  **The Engine (`AkwardJaggedAutoregressiveSequencer`)**:
    * **Responsibility:** Pure mathematics and index logic.
    * **Solves:** *"Where does a valid window start in this jagged array?"* It handles the complexity of strict windowing, padding, and vectorized index calculation.
    * 

[Image of layered software architecture diagram]


2.  **The Drivers (`DenseArraySequencer`, `RaggedArraySequencer`)**:
    * **Responsibility:** Data adaptation and validation.
    * **Solves:** *"How do I interpret this Numpy/Torch/Awkward array?"* They normalize inputs so the Engine always sees a consistent structure (Units $\times$ Time $\times$ Features).

3.  **The Leaf Dataset (`SlidingWindowBatchDataset`)**:
    * **Responsibility:** Data slicing and modality management.
    * **Solves:** *"Get me the actual data tensor for index $i$."* It manages sequencers for specific data keys (e.g., managing the 'features' array separately from the 'targets' array).

4.  **The Orchestrator (`ContextBatchDataset`)**:
    * **Responsibility:** Task semantics and synchronization.
    * **Solves:** *"Bundle the inputs and targets together for training."* It ensures that input features and target labels remain perfectly aligned in time and delivers them in a structured dictionary expected by the model.

### Key Takeaways

* **Zero-Copy Slicing:** By using Awkward Arrays and pre-computed indices, we never duplicate the underlying data in memory. Windows are only materialized as Tensors at the very last moment in `__getitem__`.
* **Task Agnostic Core:** The core engine doesn't know about "RUL" or "Forecasting." It only knows about sequences. This allows us to build new task types (like `FaultClassification`) simply by creating a new wrapper dataset, without touching the complex slicing logic.
* **Robustness:** The system automatically handles edge cases like single-step sequences, univariate data, and irregular unit lengths, preventing common index errors during training.

This architecture allows the system to scale seamlessly from simple single-variate forecasting on a laptop to complex, multi-modal industrial RUL estimation on large clusters.