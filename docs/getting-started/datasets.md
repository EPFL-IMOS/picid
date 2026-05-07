# Dataset Setup

TabPHM covers nine datasets across four PHM/forecasting task types.
Most datasets are downloaded automatically on first use; two require a one-time manual step.

---

## Quick start

Run the setup script from the repository root to pre-download all automatically
available datasets (strongly recommended before running experiments):

```bash
bash scripts/datasets_setup.sh
```

By default datasets land in `./datasets/` (the default `paths.data_dir`).
Override the directory with the `DATA_DIR` environment variable:

```bash
DATA_DIR=/my/data bash scripts/datasets_setup.sh
```

---

## Dataset overview

| Dataset | Domain | Task | Download |
|---|---|---|---|
| UNIBO Powertools (UNIBO21) | Battery degradation | Prognostics | Auto (phmd CDN) |
| C-MAPSS via phmd (N-CMAPSS) | Turbofan engine | Prognostics | Auto (phmd CDN) |
| PRONOSTIA | Bearing degradation | Prognostics | Auto (phmd CDN) |
| XJTU-SY | Bearing degradation | Prognostics | Auto (phmd CDN) |
| PHME 2020 (PHME20) | Bearing degradation | Prog / Diag | Auto (phmd CDN) |
| HSF-15 | Hydraulic system | Prognostics | Auto (phmd CDN) |
| ThreeW | Oil well operations | Anomaly detection | Auto (git sparse clone) |
| NB14 | Lithium-ion battery | Prognostics | Auto (phmd CDN) |
| N-CMAPSS (concepts variant) | Turbofan engine | Prognostics | **Manual** (NASA) |
| MZVAV | HVAC building system | Diagnostics | **Manual** (ASHRAE) |

---

## Auto-downloaded datasets

The seven auto-download datasets are fetched by the phmd library from a public CDN
(`filedn.eu`) the first time an experiment loads them.
`scripts/datasets_setup.sh` pre-fetches them so the first experiment run is not slow.

The default cache directories match the Hydra path config:

| Dataset | Cache location |
|---|---|
| PHME20, PRONOSTIA, XJTU-SY, HSF15, N-CMAPSS (phmd) | `${paths.data_dir}/phmd_cache/` |
| NB14, UNIBO21 | `${paths.data_dir}/datasets/<name>/` |

ThreeW is fetched via a sparse git clone from a GitHub mirror when the ThreeW
datasource is first instantiated. No extra configuration is needed.

---

## Manual downloads

### N-CMAPSS (concepts variant)

Used by the `concepts_n_cmapss` datasource family (experiments under
`configs/experiment/concepts_n_cmapss/`).
These experiments read raw NASA HDF5 files rather than the phmd-packaged version.

**Download:**
Go to the NASA PCoE data repository and search for **N-CMAPSS**:

> <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>

Or search for "N-CMAPSS Dataset DS01-DS08" on [data.nasa.gov](https://data.nasa.gov).

**Placement:** copy the `.h5` files into `${paths.data_dir}/N-CMAPSS/`:

```
datasets/
  N-CMAPSS/
    N-CMAPSS_DS01.h5
    N-CMAPSS_DS02.h5   ← required for most concepts experiments
    N-CMAPSS_DS03.h5   ← required for DS03 experiments
    ...
```

The default `concepts_n_cmapss` config uses DS02. Check individual experiment
configs under `configs/experiment/concepts_n_cmapss/` for the `n_DS` value.

### MZVAV

Used by experiments under `configs/experiment/mzvav/`.
MZVAV is the ASHRAE Research Project 1312-RP Multi-Zone VAV air handling unit
dataset and is not available on a public CDN.

**Source:** ASHRAE Research Project 1312-RP.
Contact your institution's ASHRAE resource or the dataset authors for access.

**Placement:** place the CSV at `${paths.data_dir}/building/MZVAV-2-2.csv`:

```
datasets/
  building/
    MZVAV-2-2.csv
```

---

## Verifying the setup

After running the setup script and placing any manual files, verify each dataset
loads correctly by running the corresponding tutorial script:

```bash
# PHME20
uv run python tutorials/datasources/phme20_loader.py

# UNIBO21
uv run python tutorials/datasources/unibo21_bosello_loader.py

# NB14
uv run python tutorials/datasources/nb14_bosello_loader.py

# N-CMAPSS concepts (requires manual .h5 files)
uv run python tutorials/datasources/concepts_n_cmapss_ds02_loader.py \
    --data-dir datasets
```

A successful run prints a split overview table with shapes and feature stats.
Any missing file will raise a `FileNotFoundError` with the expected path.

---

## Paths configuration

All dataset paths resolve through Hydra's `paths` config group.
Copy `configs/paths/default.yaml` and set `data_dir` to point at your local
dataset root, then pass `paths=<your_config>` on every experiment run:

```bash
python picid/run.py paths=my_paths experiment=phme20/prognostics/raw/xgboost_fit_predict
```

See [Install](install.md) for the full paths setup.
