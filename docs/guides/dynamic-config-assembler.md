# Dynamic Config Assembler

The **Dynamic Config Assembler** is an interactive CLI that helps you build run commands by selecting task, dataset, and model through a guided funnel. It shows which configs are used, previews the composed config, and outputs a ready-to-copy command.

## Usage

Run the assembler from the project root:

```bash
uv run picid-assemble
```

The tool prompts you interactively. You can cancel at any step by pressing Ctrl+C or choosing to exit.

## Entry points

You can assemble your experiment in three ways:

| Entry point | Flow | Best for |
|-------------|------|----------|
| **Task first** | task → experiment | When you know the task type (e.g. prognostics, anomaly_detection) and want to pick an experiment |
| **Dataset first** | group → task → experiment | When you know the dataset group (e.g. unibo) and want to explore tasks and experiments under it |
| **Model first** | model → experiment | When you want to run a specific model and see which experiments use it |

## Tiers

The assembler offers three complexity tiers:

| Tier | Description |
|------|--------------|
| **Easy** | Use defaults only. No path or parameter overrides. |
| **Medium** | Override paths (e.g. `paths=local`), optionally select a debug config for faster runs. |
| **Hard** | Override optimizer, learning rate, `trainer.max_epochs`, and model-specific params (e.g. `dropout_prob` for CNN). |

## Output

After you select an experiment and any overrides, the assembler shows:

1. **Tree** — Which config paths are read (task definitions, experiments).
2. **Config preview** — A YAML snippet of `task_definition`, `model`, and `datasource` from the composed config.
3. **CLI command** — A copy-paste ready command, e.g.:
   ```bash
   uv run python picid/run.py experiment=unibo/prognostics/combined/cnn_1d paths=default
   ```

Copy and run from the project root.
