# Coverage Omit Rationale

Modules excluded from coverage (see `pyproject.toml` [tool.coverage.run].omit) and why.

## CLI / Entry Points

- `picid/run.py` and other CLI entry points.
- **Reason:** Exercised by nox experiment sessions and manual runs; unit testing CLI is low ROI.

## Datasources with External I/O

- `picid/data/datasources/phmd_*.py`, `picid/data/datasources/agtf30k.py`, etc.
- **Reason:** Require large downloads or proprietary data; covered by integration tests where fixtures exist.

## Visualization / Demos

- `picid/transforms/visualization/features_target_plotter.py`, `picid/data/datasources/toy_example.py`
- **Reason:** UI/demo code; manual verification.
