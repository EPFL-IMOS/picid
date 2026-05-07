# Install

## Prerequisites

- Python 3.12+
- `uv`
- Git with submodule support

## Install Steps

```bash
git clone <repo-url>
cd TabPHM
uv sync
```

## Paths Configuration

Create a custom file under `configs/paths/` (copy `configs/paths/default.yaml`) and set:

- `root_dir`
- `data_dir`
- `output_dir`
- `cache_path`

Then run experiments with `paths=<your_paths_config_name>`.

## Dataset Setup

After installing, download the datasets:

```bash
bash scripts/datasets_setup.sh
```

Most datasets are fetched automatically. Two datasets (N-CMAPSS concepts variant and
MZVAV) require a one-time manual download — see [Dataset Setup](datasets.md) for
step-by-step instructions.

See also: [Quickstart](quickstart-5min.md).
