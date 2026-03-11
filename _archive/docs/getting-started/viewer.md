# Dataset viewer

The **dataset viewer** is a Panel-based web app for inspecting prepared data (e.g. after datasource load and transforms). It helps with debugging and understanding dataset structure and content.

**Entry point:** `picid_report/dataset_viewer.py`

## How to start

From the project root:

```bash
panel serve picid_report/dataset_viewer.py --autoreload --show --log-level debug
```

- `--autoreload` reloads the app when the source changes.
- `--show` opens the app in the default browser.
- `--log-level debug` increases log verbosity.

Ensure the project environment is activated and dependencies (including `panel`) are installed (e.g. via `uv pip install -e ".[dev]"`).
