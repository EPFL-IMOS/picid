# Contributing to PICID

Thank you for your interest in contributing. This document covers how to get set up and submit changes.

## Development Environment Setup

Follow the same setup as [docs/getting-started/setup.md](docs/getting-started/setup.md):

```bash
git submodule update --init --recursive
uv sync --group dev
uv pip install -r requirements-local.txt
```

Make sure your Git SSH key is configured for GitHub if needed (required for cloning submodules). On Windows, you may need [build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for numpoly.

## Running Tests

Quick test run (skips slow benchmarks and snapshot-based tests):

```bash
uv run pytest test/ -m 'not slow and not requires_snapshots'
```

Use `uv run pytest test/` for the full suite. Run from the project root or ensure `PROJECT_ROOT` is set.

## Code Style

Code follows project conventions. Before submitting:

- Run tests: `uv run pytest test/ -m 'not slow and not requires_snapshots'`
- Run type checking: `uv run mypy picid` (mypy is in dev dependencies)

Formatting and linting tools are configured in `pyproject.toml`. Keep changes consistent with the existing codebase.

## Pull Request Process

1. **Branch naming**: Use descriptive names, e.g. `fix/issue-description`, `feat/new-feature`, or `docs/update-readme`.

2. **What to include**:
   - A clear description of the change
   - Tests for new behavior when applicable
   - Ensure the standard test command passes before opening a PR

3. Open a PR against the default branch. Maintainers will review.

## Adding a New Model

See [docs/guides/how_to_add_a_new_model.md](docs/guides/how_to_add_a_new_model.md) for a step-by-step guide to adding model wrappers and experiment configs.
