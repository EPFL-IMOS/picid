# Install

## Prerequisites

- Python 3.12+
- `uv`
- Git with submodule support

## Install Steps

```bash
git clone <repo-url>
cd PICID
uv sync
```

## Paths Configuration

Create a custom file under `configs/paths/` (copy `configs/paths/default.yaml`) and set:

- `root_dir`
- `data_dir`
- `output_dir`
- `cache_path`

Then run experiments with `paths=<your_paths_config_name>`.

See also: [Quickstart](quickstart-5min.md).
