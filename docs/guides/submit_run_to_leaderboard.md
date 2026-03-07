# Submit Your Run to the Leaderboard

After running an experiment, you can add your results to the leaderboard so they appear on the docs site when it is rebuilt.

## Quick overview

```
run experiment → (optional) upload to Drive → add entry to leaderboard_links.yaml → rebuild site
```

---

## 1. Run an experiment

Use the [Experiment Generator](../experiment-generator/index.html) to pick a command, or run directly:

```bash
uv run python picid/run.py paths=rt_local experiment=phme20/prognostics/raw/xgboost_fit_predict
```

This creates a run dir under `artifacts/`, e.g.:
`artifacts/debug/runs/debug+phme20+.../2026-03-07_12-00-00/`

The run dir contains `REPRODUCE.md`, `config_resolved.yaml`, `run_metadata.yaml`, and `csv_logs/` with metrics.

## 2. (Optional) Upload to Google Drive

```bash
# Dry-run (no credentials needed — tests the workflow):
uv run python scripts/publish/push_run_to_gdrive.py <run_dir> --dry-run

# Real upload (requires scripts/publish/gdrive_config.yaml):
uv run python scripts/publish/push_run_to_gdrive.py <run_dir>
```

Keep the link it prints — you'll need it in the next step.

To also print the full submission message:

```bash
uv run python scripts/publish/push_run_to_gdrive.py <run_dir> --dry-run --print-message
```

## 3. Add your entry to `leaderboard_links.yaml`

Open `docs/leaderboard/leaderboard_links.yaml` and add your entry under the correct dataset:

```yaml
phme20:
  - experiment_key: phme20/prognostics/raw/xgboost_fit_predict
    submitter: your_name
    timestamp: "2026-03-07"
    gdrive_link: "https://drive.google.com/drive/folders/..."   # leave "" if no Drive upload
    metrics:
      rmse: 0.042
      mae: 0.031
    notes: "XGBoost baseline, default hyperparams"
```

Metrics are read from `csv_logs/` in your run dir — check the CSV files there.

## 4. Rebuild the docs site

```bash
uv run python scripts/experiment_generator/generate_config_map.py
uv run python scripts/leaderboard/build_leaderboard.py
uv run mkdocs build
```

Or serve locally to preview:

```bash
uv run python scripts/leaderboard/build_leaderboard.py
uv run mkdocs serve
```

Open [http://127.0.0.1:8000/leaderboard/](http://127.0.0.1:8000/leaderboard/) to see your entry.

## What the leaderboard page shows

- Ranked table per dataset (sorted by first metric, lower = better)
- Columns: rank, experiment key, submitter, date, metrics, reproduce link, notes
- Sortable columns (click headers)
- 📂 Run link → Google Drive folder if uploaded

[**Open Leaderboard →**](../leaderboard/index.html)
