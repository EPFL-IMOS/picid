# Repository Guidelines

## Project Structure & Module Organization
`picid/` is the main Python package. Core areas include `data/` for datasources, datasets, and preprocessing, `transforms/` for feature pipelines, `model/` and `baselines/` for training logic, `evaluator/` for metrics and hooks, and `cli/` for entry points. Hydra configuration lives in `configs/`, especially `experiment/`, `datasource/`, `model/`, `task_definition/`, and `paths/`. Tests mirror the package layout under `test/`; snapshot fixtures and generators live in `test/fixtures/snapshot/` and `test/scripts/snapshot/`. Use `docs/` for reference material, `tutorials/` for runnable examples, and treat `_archive/` and most notebooks as non-production references.

## Build, Test, and Development Commands
Initialize submodules before installing:

```bash
git submodule update --init --recursive
uv sync
uv pip install -e ".[dev]"
```

Common workflows:

```bash
uv run pytest test/ -m "not slow and not requires_snapshots"
uv run pytest test/ --cov=picid --cov-fail-under=74
uv run ruff check picid
uv run mypy picid
uv run nox -f test/pipeline/noxfile.py -s pipeline_snapshot
uv run python picid/run.py paths=rt_local experiment=railway_traction/autogluon_railway_fit_predict debug=default
```

## Coding Style & Naming Conventions
Target Python 3.12 with 4-space indentation. Follow `ruff-format` output and keep imports, whitespace, and line wrapping tool-driven. Use `snake_case` for modules, functions, YAML config names, and tests; use `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants. Add type hints in `picid/` when practical, and keep new Hydra configs grouped by dataset/task, for example `configs/experiment/phme20/...`.

## Testing Guidelines
Pytest is the standard framework. Place tests next to the mirrored area, for example `picid/data/cache/file_lock.py` -> `test/data/cache/test_file_lock.py`. Reuse the nearest `conftest.py` instead of duplicating fixtures. `slow` and `requires_snapshots` markers are active; snapshot changes usually require regenerating `test/fixtures/snapshot/reference/` with `uv run python test/scripts/snapshot/generate_snapshot_reference.py`. CI currently enforces `--cov-fail-under=74`; the repository test strategy targets 80%, so avoid coverage regressions.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit-style subjects such as `fix(scope): ...`, `feat: ...`, `refactor: ...`, `docs: ...`, and `chore: ...`. Keep subjects imperative and narrow in scope. There is no checked-in PR template, so include a short summary, linked issue, affected datasets/configs, and the exact validation commands you ran. Attach plots or screenshots only when docs, reports, or visual outputs changed.

## Configuration & Data Tips
Keep machine-specific paths in `configs/paths/*.yaml`; start from `configs/paths/default.yaml` and do not commit secrets or local dataset locations. Datasets are external to the package and resolved through the selected path config.
