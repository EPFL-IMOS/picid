# RELEASE ROADMAP

## Release Blockers (Priority Order)

| Priority | Area | Release Blocker | Evidence (file:line) | User Impact | Release Gate (Definition of Done) |
|---|---|---|---|---|---|
| P0 | Distribution | Repository has no actual open-source license text. | `LICENSE:4` (`License: To be decided.`), `pyproject.toml:16` (`# License: to be decided`) | External users and companies cannot legally use or redistribute the library. | Select an OSI license (e.g., MIT/BSD/Apache-2.0), replace `LICENSE`, add `project.license` and license classifier in `pyproject.toml`. |
| P0 | Distribution / Packaging | Published package is not self-contained for documented workflows. Runtime expects repo-level `configs/` and local layout. | `pyproject.toml:6` (wheel includes only `picid`), `picid/run.py:19`, `picid/run.py:243`, `docs/getting-started/setup.md:8-23` | `pip`/wheel users cannot reliably run the primary experiment flow without cloning repo internals and local submodules. | Package runtime configs as package data (or move into package), provide stable CLI entrypoint, and verify `pip install` + `picid-run ...` works in clean env. |
| P0 | Docs | Docs entry page contains unresolved merge-conflict marker. | `docs/index.md:49` | Immediate trust hit for first-time users; signals unstable release quality. | Remove merge marker and re-run docs build check in CI. |
| P1 | Onboarding | "Quick start" is not a sub-5-minute runnable path for a new user. Requires submodules, SSH setup, local paths, and external datasets. | `docs/getting-started/setup.md:8-40`, `docs/index.md:18-25` | New users cannot quickly validate install or value; high drop-off risk. | Add a true 5-minute smoke test using tiny bundled/synthetic data, one command, and expected output artifacts. |
| P1 | API Surface / Extensibility | Core datasource behavior is hardcoded to specific dataset splits and assumptions instead of config-driven extension points. | `picid/data/datasources/unibo/loader.py:47-146`, `picid/data/datasources/unibo/loader.py:185-188`, `picid/data/datasources/MZVAV.py:123-146` | Users cannot easily plug custom degradation datasets/splits without editing library code. | Move split definitions and label-mapping rules into config; support user-provided split strategies and metadata mapping hooks. |
| P1 | Reliability | CSV schema robustness is weak: loaders directly index expected headers without explicit validation; tests do not cover corrupted/missing header scenarios. | `picid/data/datasources/ETTh.py:55-60`, `picid/data/datasources/railway.py:56-61`, `test/data/datasources/test_etth.py:24-91`, `test/data/datasources/test_railway.py:25-63` | Real-world dirty CSVs fail with low-context pandas errors; lowers trust in data ingestion reliability. | Add schema-validation layer with actionable errors; add tests for missing timestamp/target columns, renamed headers, and malformed CSV headers. |
| P1 | API Docs | Public-facing classes in core modules are missing docstrings at scale (54 in `data/transforms/pipeline` paths), degrading generated API docs. | Examples: `picid/data/datasources/ETTh.py:8`, `picid/data/datasources/railway.py:12`, `picid/data/datasources/unibo/loader.py:149`, `picid/data/datasets/context_dataset.py:17` | Extension authors lack clear contracts and expected inputs/outputs for key classes. | Add class-level docstrings for all public classes in `picid/data`, `picid/transforms`, and `picid/pipeline`; enforce docstring linting in CI. |
| P2 | API Ergonomics | Signature/type inconsistencies reduce API clarity for extension authors. | `picid/data/datasources/ETTh.py:14` (`target_name: tuple[str, list]` but tests pass string at `test/data/datasources/test_etth.py:20`), `picid/data/datasources/base/interfaces.py:51-52` (`kwargs[...]` required keys) | Increases integration friction and confusing errors when implementing custom loaders. | Normalize type hints and constructor signatures; raise explicit `ValueError`/`TypeError` instead of indirect `KeyError` for missing required args. |

## Robustness Coverage Snapshot (Current Strength)

| Check | Status | Evidence |
|---|---|---|
| Empty dataset/empty batch edge cases | Covered | `test/data/test_edge_cases.py:21-37`, `test/evaluator/test_edge_cases.py` |
| Mismatched dimensions / shape mismatch | Covered | `test/data/datasets/test_sliding_window_batch_dataset.py:88-94`, `test/evaluator/test_forecaster_evaluator.py:129-155` |
| Corrupted CSV headers / schema drift | Missing | No dedicated negative tests in datasource test suite (`test/data/datasources/test_etth.py`, `test/data/datasources/test_railway.py`) |

## Recommended Release Sequence

1. Close all P0 blockers before any v1.0 tagging.
2. Close P1 blockers in same release cycle to ensure users can onboard, extend, and trust results.
3. Treat P2 as cleanup, but include if schedule allows because it reduces support burden post-release.
