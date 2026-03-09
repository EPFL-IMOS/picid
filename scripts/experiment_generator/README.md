# Experiment Generator

Scripts and assets for the Hydra Experiment Generator on the docs site.

## Scripts

- `generate_config_map.py` — Delegates to `picid.cli.generate_web` to scan `configs/` (same logic as `picid-assemble`) and output `config_map.json` into `docs/experiment-generator/`.

## Usage

From project root:

```bash
uv run picid-generate-web
```

Or:

```bash
uv run python scripts/experiment_generator/generate_config_map.py
```

Run this before `mkdocs build` so the docs site stays in sync with the config tree. The output `docs/experiment-generator/config_map.json` is consumed by the Experiment Generator UI.
