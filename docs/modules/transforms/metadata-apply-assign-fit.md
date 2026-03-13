# Metadata: `apply_to`, `assign_to`, `fit_on`

## `apply_to`
Selects source key(s) to read.

## `assign_to`
Controls destination key(s); omitted means in-place overwrite.

## `fit_on`
Constrains fitting split (typically `train`) to prevent leakage.

Recommended default for stateful transforms: `fit_on: train`.
