# PICID Test Strategy

## Overview

Tests ensure correctness, performance, and security of the PHM/forecasting framework for production use.

## Test Types

| Type | Location | Run Command | CI |
|------|----------|-------------|-----|
| Unit | `test/` (mirrors `picid/`) | `pytest test/ -m "not slow"` | Yes |
| Integration | `test/data/`, `test/evaluator/` | Same | Yes |
| Pipeline snapshot | `test/pipeline/test_pipeline_synthetic_snapshots.py` | `nox -f test/pipeline/noxfile.py -s pipeline_snapshot` | Optional |
| Slow/Perf | `test/data/test_performance_benchmarks.py` | `pytest -m slow` | No (nightly) |
| Security | `test/security/` | `pytest test/security/` | Yes |

**Pipeline snapshot:** Runs full pipeline on file-based synthetic data (prognostics, diagnostics, anomaly_detection, forecasting). Configs in `test/configs/`, fixtures in `test/fixtures/snapshot/`. No downloads. Compares test metrics to committed reference at 4-decimal precision; fails if pipeline behavior changes unintentionally. Regenerate reference with `test/scripts/snapshot/generate_snapshot_reference.py`.

## Coverage

- **Target:** 80% for `picid/` (enforced in CI)
- **Omit:** See `pyproject.toml` [tool.coverage.run].omit
- **Rationale:** CLI, datasource loaders with external I/O, and experimental modules excluded; covered by integration/experiment runs.

## Running Tests

```bash
# Full suite (excludes slow)
uv run pytest test/ -m "not slow" --cov=picid --cov-fail-under=80

# Via nox
nox -s tests

# Slow benchmarks only
uv run pytest test/ -m slow

# Pipeline snapshot
nox -f test/pipeline/noxfile.py -s pipeline_snapshot
```

## PHM-Specific Testing

- RUL normalized [0, 1]; fixtures in `test/data/conftest.py`, `test/evaluator/conftest.py`
- Edge cases: empty batches, perfect predictions, extreme values, many units (see `test/evaluator/test_edge_cases.py`)
