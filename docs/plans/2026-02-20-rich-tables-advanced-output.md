# Rich Tables and Advanced Output — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Use Rich for tables and other advanced console output so diagnostics (especially transform error details and pipeline error context) are readable and consistent, while keeping standard logging for log files and tracebacks.

**Architecture:** (1) Add a small Rich-based diagnostic helper in `picid.utils.rich_output` (or keep in utils) that builds Table/Panel renderables. (2) Refactor `_log_transform_error` to emit a Rich table for “first segment contents,” a Panel or Rule for the block, and still call `logger.error(..., exc_info=True)` so file handlers get the traceback. (3) Optionally refactor the pipeline `on_transform_error` hook to print a Rich table of context when available. All Rich output goes to `rich.console.Console(stderr=True)` so it appears in terminal; logging continues to use the standard logger for persistence and `exc_info`.

**Tech Stack:** Python 3.12, `rich>=14.0.0` (already in pyproject.toml), existing `picid.utils.rich_output` (Table, Tree, Console) and `picid.utils.rich_utils` (Tree, Syntax).

---

## Scope

| Location | Current behavior | Change |
|----------|------------------|--------|
| `picid/transforms/base/multisource/utils.py` — `_log_transform_error` | Many `logger.error(sep)`, `logger.error("key=... type=... %s", ...)` lines | Rich Panel + Table for segment contents, Table or grouped text for flags/case; one `logger.error` with exc_info for traceback |
| `picid/transforms/base/hooks.py` — `on_transform_error_enrich_log` | `logging.getLogger(...).error("Pipeline failed: %s", ..., extra=extra)` | Keep logger.error for message + exc_info; add optional Rich table print of `extra` when console is available |
| Elsewhere | Various `logger.error`/`print` | No change in this plan; only the two call sites above and shared helpers. |

---

## Task 1: Add Rich transform-error renderable helper

**Files:**
- Create or extend: `picid/utils/rich_output.py` (add function that builds renderables)
- Modify: `picid/transforms/base/multisource/utils.py` (import and use it)

**Step 1: Add helper in rich_output.py**

Add a function that builds Rich renderables for transform error diagnostics (no side effects):

```python
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

def build_transform_error_renderables(
    transform_name: str,
    flags: dict,
    metadata: dict,
    first_segment_keys: list,
    first_segment_rows: list[tuple[str, str, str]],  # (key, type_name, describe_data_type)
    case_analysis_line: str,
) -> list:
    """Build list of Rich renderables for transform_multi_source error diagnostics."""
    out = []
    out.append(Rule(title=f"Transform Error — {transform_name}.transform_multi_source()", style="red"))
    flags_table = Table(title="Data type flags")
    flags_table.add_column("Flag", style="cyan")
    flags_table.add_column("Value", style="yellow")
    for k, v in flags.items():
        flags_table.add_row(k, str(v))
    out.append(flags_table)
    out.append(Panel(str(metadata), title="Metadata", border_style="dim"))
    if first_segment_keys is not None and first_segment_rows:
        seg_table = Table(title="First segment contents")
        seg_table.add_column("key", style="cyan", no_wrap=True)
        seg_table.add_column("type", style="magenta")
        seg_table.add_column("description", style="green", overflow="fold")
        for row in first_segment_rows:
            seg_table.add_row(*row)
        out.append(seg_table)
    out.append(Panel(case_analysis_line, title="Case analysis", border_style="blue"))
    out.append(Rule(style="red"))
    return out
```

Use `list` return type so callers can append traceback or other renderables if needed. Adjust imports at top of `rich_output.py` (Panel, Rule) if not present.

**Step 2: Run existing rich_output tests**

```bash
pytest test/utils/test_rich_output.py -v
```

Expected: PASS (new function is additive).

**Step 3: Commit**

```bash
git add picid/utils/rich_output.py
git commit -m "feat(rich): add build_transform_error_renderables for diagnostics"
```

---

## Task 2: Refactor _log_transform_error to use Rich table and Panel

**Files:**
- Modify: `picid/transforms/base/multisource/utils.py` (lines 108–159)
- Test: add or extend test in `test/transforms/base/multisource/` or `test/utils/` to capture that _log_transform_error runs and optionally that Rich output is produced

**Step 1: Implement refactor in utils.py**

- Keep `_build_flags(data_kind, capability)` and all existing logic for `flags`, `_seg`, and case analysis.
- Build `first_segment_rows`: list of `(key, type(v).__name__, describe_data_type(v, calculate_stat=True))` for each `k, v` in `_seg.items()` when `_seg` is not None.
- Determine `case_analysis_line` string from the same if/elif as today (dense+direct, ragged+block-by-block, ragged+direct, unknown).
- Call `build_transform_error_renderables(transform_instance.__class__.__name__, flags, metadata, list(_seg.keys()) if _seg else None, first_segment_rows, case_analysis_line)` to get renderables.
- Use `Console(stderr=True).print(renderable)` for each renderable (or print a Group if Rich supports it), so terminal users see tables/panels.
- After Rich output, call one `logger.error(..., exc_info=True)` so log files get the traceback. The current code does not receive the exception object in `_log_transform_error`; it’s called from mixins.py in an `except` block just before `raise`. So `exc_info=True` will capture the traceback of the exception that’s about to be re-raised. Use a single line like:

```python
logger.error(
    "Transform error in %s (data_kind=%s, capability=%s); see stderr for Rich diagnostics.",
    t_name, data_kind, capability,
    exc_info=True,
)
```

**Step 2: Add minimal test**

In `test/transforms/base/multisource/test_utils.py` (create file if missing):

- Test that `_log_transform_error(self_mock, [segment], metadata, data_kind, capability)` runs without raising when `segment` is a small NamedTransformInput (e.g. one key, small array). Capture stderr (e.g. `Console(file=io.StringIO())` in the helper or mock Console.print) so that Rich output doesn’t clutter test runs; alternatively, just call the function and assert no exception.

Example (if using a capture pattern):

```python
def test_log_transform_error_runs_without_raise():
    import io
    from unittest.mock import MagicMock
    from picid.data.data_objects.data import NamedTransformInput
    from picid.transforms.base.multisource.utils import _log_transform_error
    segment = NamedTransformInput(x=np.zeros((2, 3)))
    meta = {"apply_to_keys": ["x"]}
    mock_transform = MagicMock(__class__=MagicMock(__name__="FakeTransform"))
    # Should not raise; may log and print
    _log_transform_error(mock_transform, [segment], meta, "dense", "dense")
```

If Rich is printed to stderr, the test still passes; optional: redirect Console to StringIO inside the function under test for cleaner test output.

**Step 3: Run tests**

```bash
pytest test/transforms/base/multisource/ -v
pytest picid/transforms/base/multisource/utils.py -v -k transform_error  # if test is colocated
```

Expected: PASS.

**Step 4: Commit**

```bash
git add picid/transforms/base/multisource/utils.py test/transforms/base/multisource/
git commit -m "refactor(transforms): use Rich table and Panel in _log_transform_error"
```

---

## Task 3: Optional — Pipeline on_transform_error hook with Rich table

**Files:**
- Modify: `picid/transforms/base/hooks.py`

**Step 1: Build and print a small Rich table for extra context**

In `on_transform_error_enrich_log`, after the existing `extra = {...}` construction, build a Rich Table with columns e.g. "Key" and "Value", add one row per `extra` item, then `Console(stderr=True).print(table)`. Keep the existing `logging.getLogger(...).error("Pipeline failed: %s", context.error, exc_info=True)` so log files still get the message and traceback.

**Step 2: Run tests**

```bash
pytest test/transforms/base/ -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git add picid/transforms/base/hooks.py
git commit -m "feat(transforms): print Rich table of context in on_transform_error hook"
```

---

## Task 4: Verification and docs

**Files:**
- Modify: `docs/transforms/HOOKS_EXAMPLES.md` or developer_guide (optional) — one sentence that error diagnostics use Rich when stderr is a terminal.

**Step 1: Manual smoke test**

Run an experiment that triggers a transform error (e.g. wrong apply_to_keys or data kind) and confirm terminal shows Rich tables/panels and log file still contains the traceback.

**Step 2: Optional doc line**

Add to Hooks or developer docs: “On transform or pipeline errors, the code prints Rich tables and panels to stderr for diagnostics; the same events are also logged with traceback for log files.”

**Step 3: Commit**

```bash
git add docs/
git commit -m "docs: note Rich diagnostics on transform/pipeline errors"
```

---

## Other places (out of scope for this plan)

- **Handlers** (`picid/transforms/base/handlers/handlers.py`): Long `KeyError`/assert messages (e.g. “Keys: …, Lengths: …”) could later be formatted as small Rich tables; left for a follow-up.
- **General logger.error across picid**: No change; only the two call sites above and the shared `build_transform_error_renderables` helper.

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-02-20-rich-tables-advanced-output.md`.

**Two execution options:**

1. **Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Parallel Session (separate)** — Open a new session with executing-plans and run the plan task-by-task with checkpoints.

Which approach?

- If **Subagent-Driven** is chosen: use superpowers:subagent-driven-development; stay in this session; one subagent per task with code review between tasks.
- If **Parallel Session** is chosen: open the plan in a worktree and use superpowers:executing-plans in the new session.
