# Model Types

This page describes the **model types** supported by the TabPHM pipeline and how they relate to data shapes and training flow. It also notes current limitations and open design points for certain families of models (e.g. foundation-style forecasters).

---

## Supported model types

The framework supports three broad ways in which models consume data. The **batch keys** your model receives (e.g. `batch.context.features_seq_x`, `batch.target.rul_seq_x`) are determined by the dataset type and config; see [Datasets](datasets.md) for the exact structure.

### 1. Fit–predict (scikit-learn style)

Models that process the **full dataset per task** in one shot: fit on train, predict on val/test. Data is presented as tabular (or flattened) arrays per task.

- **Data shape (per task):**
  - Context/input: `(1, N_samples, F)` or equivalent tabular `(N_samples, F)`
  - Target: `(1, N_samples, T)` or `(N_samples, T)`
- **Pipeline:** Uses `FitPredictDataset` (or task-wise iteration). The wrapper implements `fit(X, y)` and `predict(X)`; the pipeline calls these once per task (e.g. per unit or per horizon).
- **Examples:** AutoGluon, TabPFN, TabDPT, Carte, XGBoost (fit-predict wrappers in `picid.model.wrappers`).

### 2. Batchwise forward (contextual / windowed)

Models that consume **batched windows** at each step. Each batch has a fixed sequence length and optional prediction horizon; context and target are aligned in time.

- **Data shape (per batch):**
  - Target history: `(B, T_in, F)`
  - Target future: `(B, T_out, F)`
  - Context history: `(B, T_in, F)`
  - Context future: `(B, T_out, F)`
- **Pipeline:** Uses `ContextBatchDataset` or `RULContextBatchDataset` (via `HydraConcatDataset` + `SlidingWindowBatchDataset`). Training is typically Lightning-based with a feed-forward wrapper.
- **Examples:** MLP, CNN1D, LSTM, PatchTST, TimeLLM, batch-context XGBoost/TabPFN, statistical baselines, autoencoder.

### 3. Full-batch windowed (under discussion)

A variant where the **entire dataset** is presented as one large windowed view (no minibatches). Shapes are like batchwise but with `N_samples` instead of `B`. Performance and memory implications are non-trivial; support is unclear and may be experimental.

- **Data shape (conceptual):**
  - Target history: `(N_samples, T_in, F)`, etc.
- **Status:** Not fully standardized in the current pipeline; use with care.

---

## Foundation-style forecasters (e.g. Chronos, TimesFM, Moirai)

Some models are designed to **take as much context as possible** (train + test up to a point) and **output a single forecast** for the next \(k\) steps:

```text
|---- train ----|--- test ---|
      |---- context -----| -> |-- forecast --|
```

- **Desired I/O (conceptual):**
  - Input: long context series, e.g. `X_train (N_train, F)`, `X_test (N_test, F)` or combined.
  - Output: `y_test (N_samples, horizon, F)`.

**Current limitations and open points:**

1. **Train vs test data:** The pipeline separates train and test phases; foundation models often need both train and test context at prediction time. Fit-predict flows can be adapted (e.g. pass cached train context), but tabularizers or pre-processing may need to be optional or configurable.
2. **Windowing:** Existing datasets produce windowed batches; these models may need different re-windowing or a dedicated data path.
3. **Training:** Many of these models are heavily wrapped (e.g. Hugging Face); training may not go through Lightning in the same way as the rest of the benchmark.
4. **Regression:** Mapping regression (e.g. RUL) onto these forecasters may require extra design (target representation, loss).

**Possible direction (sketch):** For fit-predict-style integration: load train context (e.g. last part of training set) and test context from cache; use something like `DenseArraySequencer` in the predict loop for windowing if needed. This is not yet fully implemented.

---

## Testing behaviour: rolling vs static context

For forecasting evaluation, two patterns are relevant:

- **Rolling context:** The context window moves along the test series; at each step the model sees a fixed-length context and predicts the next horizon. Multiple steps yield multiple forecasts along time (fair evaluation).
- **Static context:** One long context is used to predict the entire test horizon. Simpler but not a fair comparison if other baselines use rolling context.

The pipeline and configs determine which behaviour is used for a given experiment.

---

## Where implementations live

- **Wrappers (unified interface):** `picid.model.wrappers` — all model wrappers (fit-predict and feed-forward) and the abstract bases `AbstractFitPredictWrapper`, `AbstractFeedForwardWrapper`, `AbstractFeedForwardTrainingWrapper`.
- **Baselines (backbone implementations):** `picid.baselines` — e.g. LSTM, PatchTST, Crossformer, TiDE, SpacetimeFormer, timeseries transformer, forecaster orchestration.
- **Task types:** `picid.baselines.definitions` — `REGRESSION_TASKS`, `CLASSIFICATION_TASKS`, `FORECASTING_TASKS`, `ALL_TASKS`; used by wrappers and evaluators.

To add a new model type or wrapper, see [Adding a new model](../guides/how_to_add_a_new_model.md).
