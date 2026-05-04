# The Prognostics, Health Management (PHM) and Forecasting Benchmark Arena

[![codecov](https://codecov.io/gh/EPFL-IMOS/TabPHM/branch/main/graph/badge.svg)](https://codecov.io/gh/EPFL-IMOS/TabPHM)
[![docs](https://img.shields.io/badge/docs-available-brightgreen)](docs/index.md)

## Getting started

Full documentation is available in [`docs/`](docs/index.md).

## Quickstart

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Git with submodule support

### Install

```bash
git clone <repo-url>
cd TabPHM
git submodule update --init --recursive
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
uv pip install -e ".[dev]"
```

### Configure paths

Copy `configs/paths/default.yaml` to a new file (e.g. `configs/paths/local.yaml`) and set:

```yaml
root_dir: /path/to/project
data_dir: /path/to/data
output_dir: /path/to/outputs
cache_path: /path/to/cache
```

### Run your first experiment

```bash
uv run picid-run paths=local debug=default experiment=unibo/prognostics/raw/cnn_1d
```

For full configuration options see the [documentation](docs/getting-started/install.md).

## Documentation

To browse the full docs locally:

```bash
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).