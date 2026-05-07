# Datasource tutorials

Runnable scripts under this directory load a datasource (or a Hydra-composed config), print split overviews, and reuse the shared standard report sections from `_standard_report.py`.

Run from the repository root so `configs/` and imports resolve:

```bash
cd /path/to/PICID
python tutorials/datasources/<script>.py --help
```

## Common CLI flags

Most loader tutorials accept:

- **`--data-dir`** — Override the PICID datasets root (`paths.data_dir`). Hydra-based tutorials forward this as a `paths.data_dir=...` override; others resolve files or PHMD cache under that root per their YAML.
- **`--no-download`** — Only on loaders that support disabling automatic fetch (**Airbus Helicopter**, **3W**). Other tutorials rely on PHMD or local layout without this toggle.

## Loader scripts (regression matrix)

| Script | Notes |
| --- | --- |
| `airbus_helicopter_loader.py` | Local HDF5 layout; `--no-download` supported. |
| `concepts_n_cmapss_ds02_loader.py` | Hydra multisource DS02 preset; `--data-dir` sets N-CMAPSS root via `paths.data_dir`. |
| `hsf15_accumulator_loader.py` | Thin wrapper; same flags as other `hsf15_*` wrappers. |
| `hsf15_component_loader.py` | Requires **`--component`** (`accumulator`, `cooler`, `pump`, `valve`). |
| `hsf15_cooler_loader.py` | Fixed component wrapper. |
| `hsf15_pump_loader.py` | Fixed component wrapper. |
| `hsf15_valve_loader.py` | Fixed component wrapper. |
| `nb14_bosello_loader.py` | NB14 Bosello splits; `--data-dir` for dataset root. |
| `phme20_loader.py` | Hydra `datasource=phme20`; PHMD cache under `${paths.data_dir}/phmd_cache`. |
| `threew_loader.py` | 3W exploration / ragged stats; extra flags for folds; `--no-download` supported. |
| `unibo21_bosello_loader.py` | UNIBO21 Bosello splits; `--data-dir` for dataset root. |
| `xjtu_sy_loader.py` | Hydra `datasource=xjtu_sy`; PHMD cache under `${paths.data_dir}/phmd_cache`. |

The list above is kept in sync with `STANDARD_DATASOURCE_TUTORIAL_SCRIPTS` in `_tutorial_cli.py` (used by `test/tutorials/test_datasource_tutorial_contract.py`).

## Other examples

- `01_load_toy.py`, `02_single_vs_multi.py` — Earlier walkthroughs; not part of the standard loader CLI matrix.

## Internals

- `_loader_introspection.py` — Record flattening, ragged summaries, metadata helpers.
- `_standard_report.py` — Shared section titles and report builders.
- `_tutorial_cli.py` — Shared `--data-dir` / `--no-download` argparse helpers and the script registry for tests.
