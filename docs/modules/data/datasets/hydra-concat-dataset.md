# HydraConcatDataset

`HydraConcatDataset` orchestrates per-unit dataset construction and exposes a single iterable surface.

## Responsibilities

- instantiates one child dataset per unit
- preserves per-unit boundaries
- supports vectorized/non-vectorized concat behavior based on dataset type

This is the bridge between split-keyed container output and model-facing batch iteration.
