# Run Lifecycle

`picid/run.py` executes the canonical sequence:

1. load composed Hydra config
2. instantiate datasource + transforms manager
3. preprocess data (direct or cached)
4. build datasets + datamodule
5. instantiate evaluators and model wrapper
6. build trainer and run `fit`/`test` path

Fit-predict and pretrained paths may skip parts of training flow and run evaluation directly.
