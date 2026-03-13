# Loss, Optimizer, and LR Scheduler

Optimization components are instantiated via Hydra:

- loss modules: `picid.loss`
- optimizer modules: `picid.optimizer`
- scheduler modules: `picid.lr_scheduler`

These are wired into Lightning modules in the orchestration layer.
