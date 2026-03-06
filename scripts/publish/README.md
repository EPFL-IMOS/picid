# Publish Scripts

Scripts for packaging run results and submitting them to the leaderboard.

## Scripts

- `build_submission_message.py` — Reads a run dir and builds a self-contained submission message (REPRODUCE.md + run_metadata.yaml + optional Drive link).
- `push_run_to_gdrive.py` *(planned)* — Uploads a run dir to Google Drive and returns a shareable link.

## Usage

### Build submission message

```bash
uv run python scripts/publish/build_submission_message.py <run_dir> [--gdrive-link URL] [--output FILE]
```

Example:

```bash
uv run python scripts/publish/build_submission_message.py \
  artifacts/debug/runs/debug+phme20+prognostics+raw+xgboost_fit_predict/2026-03-07_12-00-00 \
  --gdrive-link "https://drive.google.com/drive/folders/YOUR_FOLDER_ID"
```

Prints the submission message to stdout, or writes it to `--output`.

## Workflow

1. Run experiment → run dir under `artifacts/`
2. *(once push_run_to_gdrive is implemented)* Upload: `push_run_to_gdrive <run_dir>` → Drive link
3. Build message: `build_submission_message <run_dir> --gdrive-link <link>`
4. Submit message to the leaderboard
