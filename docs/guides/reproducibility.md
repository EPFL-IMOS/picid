# Reproducibility

How to reproduce experiments, compare configs, and switch environments.

---

## Run and reproduce

Each run writes a **REPRODUCE.md** in its output directory with two options:

- **Option A – from repo configs**: Run with Hydra overrides (uses current configs).
- **Option B – from run config**: Run `scripts/reproducibility/reproduce_from_run.py <run_dir>` to use the exact config from that run (for version comparison).

See `tutorials/workflow/reproduce_experiment.py` (runnable script).

---

## Compare configs between experiments

When metrics differ between runs, compare resolved configs to find what changed:

```bash
uv run python scripts/reproducibility/compare_configs.py run_a/config_resolved.yaml run_b/config_resolved.yaml
```

Output: `-` only in first, `+` only in second, `~` different values.

See `scripts/reproducibility/README.md` and `tutorials/workflow/compare_configs.py` (runnable walkthrough).

---

## Git version in run report

Each run records the **git commit** for reproducibility. Check `run_metadata.yaml` in the run output dir:

```yaml
git_commit: abc123def
git_branch: main
git_dirty: false
```

Use this to checkout the exact code version when reproducing.

---

## Switching uv.lock for different environments

To run with different dependency sets (e.g. CPU vs CUDA):

1. **Copy a lock file** into place, then sync:
   ```bash
   cp uv.lock.cuda uv.lock && uv sync
   ```

2. **Or use a run's uv.lock** (exact versions from that run):
   ```bash
   cp artifacts/.../run_id/uv.lock . && uv sync
   ```

See `tutorials/workflow/uv_lock_switching.py` (runnable script).
