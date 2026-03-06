# Experiment Generator

Scripts and assets for the Hydra Experiment Generator on the docs site.

## Scripts

- `generate_config_map.py` — Crawls `configs/` and outputs `config_map.json` into `docs/experiment-generator/`.

## Usage

From project root:

```bash
uv run python scripts/experiment_generator/generate_config_map.py
```

Run this after adding or changing configs under `configs/`. The output `docs/experiment-generator/config_map.json` is consumed by the Experiment Generator UI.
