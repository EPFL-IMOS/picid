# Framework Rename: lmetk → picid (History-Preserving) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename the core framework directories (`lmetk` → `picid`, `lmetk_report` → `picid_report`) and all internal references while preserving 100% of Git history, delivered as two distinct commits.

**Architecture:** Phase 1 performs directory renames only via `git mv` and one commit so Git records renames and history stays intact. Phase 2 updates all Python imports, config `_target_` strings, build/config paths, and docs in a second commit. Phase 3 validates history and tests.

**Tech Stack:** Git, Python, setuptools (pyproject.toml), pytest, Hydra/OmegaConf.

---

## Scope Summary

- **Directories:** `./lmetk` → `./picid`, `./lmetk_report` → `./picid_report`.
- **Import/string replacements:** `lmetk` → `picid`, `lmetk_report` → `picid_report` everywhere (code, configs, docs, scripts, notebooks, .gitignore, codecov, pyproject).
- **Project name in pyproject.toml:** Currently `name = "TabPHM"`. Changing to `picid` is optional (affects installable name); plan includes it as an explicit step—implementer may keep `TabPHM` if preferred.
- **No content changes in Phase 1**—only moves.

---

## PHASE 1: Structural Rename (History Preservation)

### Task 1.1: Move core directories with Git

**Files:** N/A (directory moves only)

**Step 1: Rename main package directory**

```bash
git mv lmetk picid
```

**Step 2: Rename report package directory**

```bash
git mv lmetk_report picid_report
```

**Step 3: Verify no file contents were changed**

```bash
git status
git diff --staged --stat
```

Expected: Only renames (renames shown as delete + add with rename detection). No line changes in any file.

**Step 4: Commit Phase 1**

```bash
git add -A
git commit -m "chore: rename framework directories from lmetk to picid"
```

Do not modify any file contents in this commit. This keeps Git rename detection at 100% for `git log --follow`.

---

## PHASE 2: Global Internal Reference Update

All edits below assume Phase 1 is committed (directories are already `picid` and `picid_report`).

### Task 2.1: Python imports and string literals (lmetk → picid)

**Files:** All under `picid/`, `test/` (excluding `test/lmetk_report` which will be renamed in 2.2), `notebooks/`, `pseudo_run.py`, and any script that imports or references `lmetk`.

**Step 1: Replace in Python files**

- Replace `from lmetk` → `from picid`
- Replace `import lmetk` → `import picid`
- Replace `lmetk.` (e.g. in docstrings or strings) → `picid.`
- Replace string literals containing `lmetk` (e.g. logger names, provider names) → `picid`

**Key locations (grep for `lmetk` to find all):**

- `picid/**/*.py` (all imports are now under `picid/`, so every internal `lmetk` reference)
- `test/**/*.py` (except under `test/lmetk_report/` until 2.2)
- `test/utils.py` line ~19: `provider="lmetk.defaults"` → `provider="picid.defaults"`
- `test/utils/__init__.py` line ~13: `provider="lmetk.defaults"` → `provider="picid.defaults"`
- `pseudo_run.py` if it references `lmetk`

**Step 2: Replace in configs (YAML _target_ and paths)**

- In `configs/**/*.yaml`: every `_target_: lmetk.` → `_target_: picid.`

**Step 3: Run search to ensure no missed Python/config references**

```bash
rg -l 'lmetk' --glob '!*.md' .
```

Then update any remaining non-doc files (e.g. scripts, notebooks JSON cells).

### Task 2.2: Rename test package for report and update lmetk_report → picid_report

**Files:** Rename directory; then all references.

**Step 1: Rename test report directory**

```bash
git mv test/lmetk_report test/picid_report
```

**Step 2: Replace lmetk_report in code and configs**

- In **all** repo files (including under `picid_report/`, `test/picid_report/`):
  - `from lmetk_report` → `from picid_report`
  - `import lmetk_report` → `import picid_report`
  - `lmetk_report.` → `picid_report.`
  - String literals and logger names: `lmetk_report` → `picid_report`

**Step 3: Verify no remaining lmetk_report**

```bash
rg -l 'lmetk_report' .
```

Update any remaining occurrences (docs, notebooks, .gitignore).

### Task 2.3: Build and tooling config

**Files:**

- Modify: `pyproject.toml`
- Modify: `codecov.yml`
- Modify: `.gitignore`

**Step 1: Update pyproject.toml**

- `packages = ["lmetk"]` → `packages = ["picid"]`
- `[tool.coverage.run]` → `source = ["picid"]`
- `[tool.coverage.report]` and `omit` / paths: replace `lmetk/` with `picid/` (e.g. `picid/data/datasources/picasso/...`, `picid/data/datasources/unibo/...`)

**Step 2: Optional — project name**

- If switching installable name to picid: `name = "TabPHM"` → `name = "picid"` in `[project]`. Otherwise leave as `TabPHM`.

**Step 3: Update codecov.yml**

- `paths: - lmetk/data/` → `paths: - picid/data/`

**Step 4: Update .gitignore**

- `lmetk_report_artifacts/` → `picid_report_artifacts/`
- Any path containing `lmetk` → `picid` equivalent

### Task 2.4: Docs and scripts

**Files:** `docs/**/*.md`, `scripts/**/*.sh`, `notebooks/**/*.ipynb`

**Step 1: Replace in docs**

- In `docs/`: all `lmetk` → `picid`, `lmetk_report` → `picid_report` (paths, module names, CLI examples).
- Example: `docs/concepts/from_data_to_prediction.md` — `python lmetk/run.py` → `python picid/run.py`.

**Step 2: Replace in scripts**

- In `scripts/**/*.sh`: paths like `lmetk/run.py` → `picid/run.py`, and any `lmetk_report` → `picid_report`.

**Step 3: Replace in notebooks**

- In `notebooks/**/*.ipynb`: in JSON cells, replace `lmetk` and `lmetk_report` with `picid` and `picid_report` (e.g. `lmetk.data.datasets`, `lmetk_report.run`).

### Task 2.5: Cleanup and Phase 2 commit

**Files:** Remove generated dirs; .gitignore already updated in 2.3.

**Step 1: Remove generated package metadata**

```bash
rm -rf TabPHM.egg-info lmetk.egg-info picid.egg-info 2>/dev/null || true
```

**Step 2: Verify .gitignore**

- Ensure `.gitignore` does not reference `lmetk` or old report paths; use `picid_report_artifacts/` (or similar) as needed.

**Step 3: Stage and commit Phase 2**

```bash
git add -A
git status
git diff --staged --stat
git commit -m "chore: update all references from lmetk/lmetk_report to picid/picid_report"
```

---

## PHASE 3: Validation

### Task 3.1: Verify Git history

**Step 1: Check history for a core file**

```bash
git log --oneline --follow -5 picid/run.py
```

Expected: Full history from when the file lived under `lmetk/` is visible (same commits as before rename).

**Step 2: Check report package**

```bash
git log --oneline --follow -3 picid_report/run.py
```

Expected: History preserved across rename from `lmetk_report/`.

### Task 3.2: Install and run tests

**Step 1: Editable install**

```bash
cd /Users/leone/projects/TabPHM
uv sync
uv pip install -e ".[dev]"
```

Expected: No errors; package installs as `picid` (and optionally `picid_report` if exposed as subpackage or separate package).

**Step 2: Run test suite**

```bash
uv run pytest test/ -v
```

Expected: All tests pass. If any import or config errors reference `lmetk` or `lmetk_report`, fix in Phase 2 and re-commit.

---

## DELIVERABLES

1. **Summary of changes**
   - Phase 1: Two directory renames (`lmetk` → `picid`, `lmetk_report` → `picid_report`), one commit, no file content changes.
   - Phase 2: All Python imports, config `_target_` strings, paths in pyproject.toml, codecov.yml, .gitignore, docs, scripts, notebooks updated; test directory `test/lmetk_report` → `test/picid_report`; optional project name change; cleanup of egg-info; one commit.

2. **Confirmation**
   - `git log --follow picid/run.py` (and optionally `picid_report/run.py`) shows full history.
   - `uv run pytest test/` passes.

3. **Two commits**
   - Commit 1: `chore: rename framework directories from lmetk to picid`
   - Commit 2: `chore: update all references from lmetk/lmetk_report to picid/picid_report`

---

## Quick reference: where "lmetk" appears

- **Python:** `picid/` (all), `test/` (all), `test/utils.py`, `test/utils/__init__.py`, `pseudo_run.py`, `notebooks/*.ipynb`
- **Configs:** `configs/**/*.yaml` (`_target_: lmetk.` and similar)
- **Docs:** `docs/**/*.md`
- **Scripts:** `scripts/**/*.sh`
- **Report package:** `picid_report/` (after rename), `test/picid_report/` (after rename)
- **Build/tooling:** `pyproject.toml`, `codecov.yml`, `.gitignore`

Use `rg 'lmetk'` and `rg 'lmetk_report'` before Phase 2 commit to ensure no references remain (excluding plan/docs that describe the rename itself).
