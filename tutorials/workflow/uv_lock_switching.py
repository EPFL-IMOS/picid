#!/usr/bin/env python3
"""Tutorial: Switching uv.lock for different environments.

Run experiments with different dependency sets (e.g. CPU vs CUDA) by swapping
the lock file before uv sync.

Run from project root:
    uv run python tutorials/workflow/uv_lock_switching.py

Note: uv does not support --lockfile for alternative lock files. Use copy + sync.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

print("=" * 60)
print("Tutorial: Switching uv.lock for Different Environments")
print("=" * 60)

print("""
uv uses uv.lock in the project root. To switch environments:

1. USE A RUN'S LOCK (exact versions from that run):
   cp artifacts/.../run_id/uv.lock . && uv sync

2. USE NAMED LOCK FILES (if you maintain them):
   cp uv.lock.cuda uv.lock && uv sync   # CUDA build
   cp uv.lock.cpu uv.lock && uv sync    # CPU-only

3. FROZEN SYNC (no lock changes, install from current uv.lock):
   uv sync --frozen

4. LOCKED SYNC (error if lock would change):
   uv sync --locked
""")

# Demo: show how to backup/restore (no file created by default)
uv_lock = PROJECT_ROOT / "uv.lock"
if uv_lock.exists():
    print("To backup before switching: cp uv.lock uv.lock.backup")
    print("To restore: cp uv.lock.backup uv.lock && uv sync")
else:
    print("No uv.lock found. Run uv lock first.")

print("\nDone.")
