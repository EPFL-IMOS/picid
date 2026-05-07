# Publish Scripts

Scripts for packaging run results.

## Scripts

| Script | Purpose |
|--------|---------|
| `build_submission_message.py` | Reads a run dir and builds a self-contained submission message (REPRODUCE.md + run_metadata.yaml + Drive link) |
| `push_run_to_gdrive.py` | Zips key files from a run dir and uploads to Google Drive (or `--dry-run` for testing) |
| `datasets.yaml` | Schema of supported datasets and required files per submission |
| `gdrive_config.example.yaml` | Template for Google Drive credentials config (copy to `gdrive_config.yaml`) |

## Full workflow

```
experiment → run_dir → push_run_to_gdrive → Drive link → build_submission_message → submission message
```

### 1. Run experiment

Run an experiment directly:

```bash
uv run python picid/run.py paths=<your_paths> experiment=phme20/prognostics/raw/xgboost_fit_predict
```

This creates a run dir under `artifacts/`, e.g.:
`artifacts/debug/runs/debug+phme20+prognostics+raw+xgboost_fit_predict/2026-03-07_12-00-00/`

### 2. Upload to Google Drive

**Dry run (no credentials needed):**

```bash
uv run python scripts/publish/push_run_to_gdrive.py <run_dir> --dry-run
```

**Real upload (requires `gdrive_config.yaml`):**

```bash
cp scripts/publish/gdrive_config.example.yaml scripts/publish/gdrive_config.yaml
# fill in gdrive_folder_id and credentials_path
uv run python scripts/publish/push_run_to_gdrive.py <run_dir>
```

### 3. Build submission message (optional standalone)

```bash
uv run python scripts/publish/build_submission_message.py <run_dir> --gdrive-link <link>
```

### 4. Combined: upload + print message

```bash
uv run python scripts/publish/push_run_to_gdrive.py <run_dir> --dry-run --print-message
```

Uploads (or dry-runs), then prints the full submission message with the Drive link embedded.

## Google Drive configuration

Copy the example config and fill in your values:

```bash
cp scripts/publish/gdrive_config.example.yaml scripts/publish/gdrive_config.yaml
```

`gdrive_config.yaml` is gitignored. Do not commit it.

The `push_run_to_gdrive.py` script will raise `NotImplementedError` for the actual upload until the
PyDrive2/google-api-python-client integration is implemented. Use `--dry-run` until then.

## Dataset schema

`datasets.yaml` defines which datasets are supported for submission and which files are required.
Used for future validation in `push_run_to_gdrive` and `build_submission_message`.
