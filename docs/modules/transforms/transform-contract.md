# Transform Contract

A transform has two parts:

1. **Logic block**: Python class (`_target_`) and hyperparameters.
2. **Metadata block**: routing and fit/apply semantics.

Core metadata keys:

- `apply_to`: input key(s)
- `assign_to`: output key(s)
- `fit_on`: split used for fitting stateful transforms

This contract enables reusable transform programs with no code changes.
