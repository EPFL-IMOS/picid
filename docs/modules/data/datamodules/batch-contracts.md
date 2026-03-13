# Batch Contracts

`BaseDataModule` enforces loader-level behavior:

- split-specific batching
- train shuffle default only
- optional subsetting for debug/perf workflows
- compatibility checks for supported dataset base classes

Batch schema depends on dataset type:

- RUL dataset: `features`, `rul`, optional `unit_id`
- context dataset: nested `target/context` with `*_seq_x` and `*_seq_y`
- fit-predict dataset: task-wise arrays
