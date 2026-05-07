# System Architecture

PICID is configuration-driven around Hydra. Core execution path:

`datasource -> SplitDatasetContainer -> datasets -> transforms -> datamodule -> wrapper/lightning module -> evaluator -> metrics/artifacts`

Primary module docs:

- [Data](../modules/data/index.md)
- [Transforms](../modules/transforms/index.md)
- [Modeling](../modules/modeling/index.md)
- [Orchestration](../modules/orchestration/index.md)
- [Evaluation](../modules/evaluation/index.md)
