# CLI

CLI entry points are defined in `pyproject.toml`:

- `picid-assemble` — interactive config assembler; builds a ready-to-run `picid-run` command from guided prompts.
- `picid-run` — main experiment runner (wraps `picid/run.py`); accepts Hydra overrides.

Implementation packages: `picid.cli` (assembler), `picid.run` (runner).

API: [picid.cli](../../reference/api/picid_cli.md)
