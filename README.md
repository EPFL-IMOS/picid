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

<details open>
<summary>Standard (with foundation models, from git)</summary>

Installs TabPFN, TabDPT, carte-ai and phmd directly from their git repositories.

```bash
git clone <repo-url>
cd TabPHM
uv sync
```

</details>
<details open>
<summary>Without foundation models</summary>

Lighter install — skips the TabFM and phmd groups entirely.

```bash
git clone <repo-url>
cd TabPHM
uv sync --no-group tabfm --no-group phmd
```

</details>
<details open>
<summary>phmd only (no foundation models)</summary>

Installs phmd from git but skips TabPFN / TabDPT / carte-ai.

```bash
git clone <repo-url>
cd TabPHM
uv sync --no-group tabfm
```

</details>
<details open>
<summary>Local submodule install (for developing foundation models)</summary>

Uses editable installs from local submodule checkouts instead of git. Requires submodules to be present.

```bash
git clone <repo-url>
cd TabPHM
git submodule update --init --recursive
uv sync \
  --no-group tabfm --group tabfm-local \
  --no-group phmd  --group phmd-local
```

</details>


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

Or by activating the venv first:

```bash
source .venv/bin/activate
python picid/run.py paths=local debug=default experiment=unibo/prognostics/raw/cnn_1d
```

For full configuration options see the [documentation](docs/getting-started/install.md).

## Documentation

To browse the full docs locally:

```bash
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).