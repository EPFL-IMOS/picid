# Layered Test Matrix

Run all commands from the repository root with `uv run pytest`.

## Default Review Path

For this branch, the default focused suite is:

```bash
uv run pytest test -q -m "not slow and not benchmark"
```

That keeps long-running and timing-focused checks out of the common review loop while still exercising the main correctness suite.

## Marker Meaning

| Marker | Meaning | Default review path |
|---|---|---|
| `unit` | Isolated tests with minimal dependencies and fast runtime | Included in lean lane |
| `slow` | Longer-running tests that are useful but not part of the focused loop | Excluded |
| `benchmark` | Timing and throughput checks | Excluded |
| `tutorial` | Narrative/tutorial coverage; often still safe to run locally | Included unless also marked otherwise |
| `integration` | Cross-subsystem behavior checks | Included unless also marked otherwise |
| `optional_dep` | Needs non-default dependencies or environment | Run intentionally when relevant; these may still skip noisily if your local environment lacks extras |
| `requires_snapshots` | Verification-only against committed snapshot fixtures | Never regenerate snapshots from test runs |

## Snapshot Safety

Snapshot-backed tests in this branch are verification-only.

- Do not regenerate, rewrite, bless, or update committed snapshots from test runs.
- If a snapshot test fails, treat that as a behavior change to inspect, not a fixture-refresh task.
- `requires_snapshots` remains opt-in when fixtures are unavailable.

## Layered Commands

| Layer | Command | Result on this branch |
|---|---|---|
| Lean unit lane | `uv run pytest test -q -m "unit and not optional_dep and not slow and not benchmark"` | Fast contract verification for local iteration and CI precheck |
| Collection sanity | `uv run pytest --collect-only -q test` | `2769 tests collected in 31.82s` |
| Foundational unit layer | `uv run pytest test/utils test/data/data_objects test/transforms/base -q` | `668 passed, 2 skipped, 1 warning in 11.32s` |
| Datasource and tutorial layer | `uv run pytest test/data/datasources test/tutorials -q` | `285 passed in 59.09s` |
| Evaluator and pipeline layer | `uv run pytest test/evaluator test/pipeline -q` | Included in the broader focused suite below; dedicated reruns in this session became noisy after overlapping shell invocations, so the authoritative green signal came from the full focused suite |
| Focused default suite | `uv run pytest test -q -m "not slow and not benchmark"` | `2752 passed, 10 skipped, 8 deselected by marker filter, 106 warnings in 626.15s (0:10:26)` |

## Optional Commands

Use these when you are touching the corresponding test families:

```bash
# Lean lane used by fast CI checks
uv run pytest test -q -m "unit and not optional_dep and not slow and not benchmark"

# Full non-benchmark lane
uv run pytest test -q -m "not slow and not benchmark"

# Benchmark/timing checks only
uv run pytest test/data/test_performance_benchmarks.py -q -m benchmark

# Datasource and tutorial checks
uv run pytest test/data/datasources test/tutorials -q

# Foundational utility / container / transform base checks
uv run pytest test/utils test/data/data_objects test/transforms/base -q
```

## Reviewer Guidance

For ordinary review on this branch:

1. Run collection sanity.
2. Run the focused default suite.
3. Add the narrower layer that matches the area you changed.

If you touch timing-only tests, run the benchmark command explicitly.

If you touch snapshot-backed pipeline tests, validate against the committed snapshots only and do not regenerate them.
