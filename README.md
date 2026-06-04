# The Prognostics, Health Management (PHM) and Forecasting Benchmark Arena


[![docs](https://img.shields.io/badge/docs-available-brightgreen)](docs/index.md)
[![codecov](https://codecov.io/gh/picid-research/picid/branch/main/graph/badge.svg)](https://codecov.io/gh/picid-research/picid)

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
cd picid
uv sync
```

</details>
<details open>
<summary>Without foundation models</summary>

Lighter install — skips the TabFM and phmd groups entirely.

```bash
git clone <repo-url>
cd picid
uv sync --no-group tabfm --no-group phmd
```

</details>
<details open>
<summary>phmd only (no foundation models)</summary>

Installs phmd from git but skips TabPFN / TabDPT / carte-ai.

```bash
git clone <repo-url>
cd picid
uv sync --no-group tabfm
```

</details>
<details open>
<summary>Local editable install (for developing foundation models)</summary>

Uses editable installs from local checkouts after the standard git-backed sync.

```bash
git clone <repo-url>
cd picid
bash scripts/sync_local_packages.sh
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

### Datasets

Most datasets are downloaded automatically the first time the datasource is
executed, if the data is not already present locally.
This covers: UNIBO21, NB14, PHME20, PRONOSTIA, XJTU-SY, HSF-15, N-CMAPSS
(phmd variant), and ThreeW.

> **Note:** The **N-CMAPSS (concepts variant)** datasource does not yet have
> automatic download support and requires manual placement of the raw NASA HDF5
> files. This will be added in a future update.

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


## Citation

Several works have been implemented in Picid. If you wish to cite us please use:

- [Picid: A Modular Evaluation Infrastructure for Reproducible PHM Across Tasks and Domains (05/2026)](https://arxiv.org/abs/2605.28345)
- [From paper to benchmark: agentic, framework-based reproduction of under-specified methods in machine health intelligence (05/2026)](https://arxiv.org/abs/2605.28371)
- [Towards Unified and Data-Efficient Prognostics and Health Management with Tabular Foundation Models (upcoming!)](papers/2025_TabPHM_Paper_ArXiv.pdf)


```
@article{telyatnikov2026picid,
  title={Picid: A Modular Evaluation Infrastructure for Reproducible PHM Across Tasks and Domains},
  author={Telyatnikov, Lev and Theiler, Raffael and Von Krannichfeldt, Leandro and Fink, Olga},
  journal={arXiv preprint arXiv:2605.28345},
  year={2026}
}

@article{theiler2026paper,
  title={From paper to benchmark: agentic, framework-based reproduction of under-specified methods in machine health intelligence},
  author={Theiler, Raffael and Comito, Ludovico and Leko, David and Von Krannichfeldt, Leandro and Telyatnikov, Lev and Fink, Olga},
  journal={arXiv preprint arXiv:2605.28371},
  year={2026}
}
```





## License

This project is licensed under a non-commercial license. Use is permitted for academic and government-funded research only. Commercial use is prohibited. See [LICENSE.txt](LICENSE.txt) for full terms.
