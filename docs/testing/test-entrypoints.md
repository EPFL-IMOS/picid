# Test entrypoints

## Canonical command

Run pytest **from the repository root** with the project-managed environment:

```bash
uv run pytest
```

Use `uv run pytest` so the interpreter and dependencies match `pyproject.toml` and `uv.lock`. Invoking bare `python -m pytest` or a system `pytest` can point at the wrong environment and produce misleading import or dependency errors.

## Common invocations

- Full tree, collect only (fast sanity check):

  ```bash
  uv run pytest --collect-only -q test
  ```

- Marker and collection contract meta-tests:

  ```bash
  uv run pytest test/meta/test_collection_contract.py -q
  ```

- Typical local run (see [TEST_STRATEGY.md](TEST_STRATEGY.md) for CI and markers):

  ```bash
  uv run pytest test/ -m "not slow"
  ```

Install dev dependencies first if needed: `uv sync --group dev`.
