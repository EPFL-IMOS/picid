# First Run Walkthrough

A standard run follows this flow:

1. Datasource loading and splitting
2. Preprocessing (transforms, optional cache)
3. Dataset and datamodule creation
4. Model wrapper and Lightning module instantiation
5. Training/testing + evaluator metrics

Use this command as a baseline:

```bash
python picid/run.py paths=<your_paths> experiment=<experiment_key>
```

Then inspect:

- resolved config (`config_resolved.yaml`)
- checkpoints
- evaluation artifacts
