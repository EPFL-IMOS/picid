#!/usr/bin/env bash
# datasets_setup.sh — pre-download all datasets that support automatic download.
#
# Run from the repository root:
#
#   bash scripts/datasets_setup.sh
#
# By default datasets land in ./datasets/ (the default paths.data_dir).
# Override with DATA_DIR:
#
#   DATA_DIR=/my/data bash scripts/datasets_setup.sh
#
# Datasets that require manual download are documented at the bottom of this
# file and in docs/getting-started/datasets.md.

set -euo pipefail

DATA_DIR="${DATA_DIR:-datasets}"

echo "Using DATA_DIR=$DATA_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/phmd_cache"

# ---------------------------------------------------------------------------
# Auto-downloadable datasets (via phmd CDN at filedn.eu)
# These are fetched and unpacked automatically on first experiment run, but
# pre-downloading here avoids slow first-run delays.
# ---------------------------------------------------------------------------

echo "Downloading PHME20 (bearing degradation / diagnostics)..."
uv run python -c "from phmd.download import download; download('PHME20', cache_dir='$DATA_DIR/phmd_cache', unzip=True)"

echo "Downloading PRONOSTIA (bearing degradation)..."
uv run python -c "from phmd.download import download; download('PRONOSTIA', cache_dir='$DATA_DIR/phmd_cache', unzip=True)"

echo "Downloading XJTU-SY (bearing degradation)..."
uv run python -c "from phmd.download import download; download('XJTU-SY', cache_dir='$DATA_DIR/phmd_cache', unzip=True)"

echo "Downloading HSF-15 (hydraulic system)..."
uv run python -c "from phmd.download import download; download('HSF15', cache_dir='$DATA_DIR/phmd_cache', unzip=True)"

echo "Downloading N-CMAPSS via phmd (turbofan engine / prognostics)..."
uv run python -c "from phmd.download import download; download('NCMAPSS', cache_dir='$DATA_DIR/phmd_cache', unzip=True)"

# NB14 and UNIBO21 use a different cache layout: they download directly into
# DATA_DIR/datasets/<name>/ (not DATA_DIR/phmd_cache/).
echo "Downloading NB14 (lithium-ion battery)..."
uv run python -c "from phmd.download import download; download('NB14', cache_dir='$DATA_DIR', unzip=True)"

echo "Downloading UNIBO21 (battery degradation)..."
uv run python -c "from phmd.download import download; download('UNIBO21', cache_dir='$DATA_DIR', unzip=True)"

echo ""
echo "Auto-download complete."
echo ""

# ---------------------------------------------------------------------------
# Datasets that CANNOT be downloaded automatically
# ---------------------------------------------------------------------------
#
# N-CMAPSS (concepts variant, used by concepts_n_cmapss datasource)
# -----------------------------------------------------------------
# The concept-learning experiments use raw HDF5 files published by NASA.
# Download the dataset from:
#   https://data.nasa.gov/dataset/N-CMAPSS-Dataset-DS01-DS08
# or search for "N-CMAPSS" on the NASA PCoE data repository:
#   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
#
# Place the files at:
#   $DATA_DIR/N-CMAPSS/N-CMAPSS_DS01.h5
#   $DATA_DIR/N-CMAPSS/N-CMAPSS_DS02.h5
#   ...  (up to DS08, depending on which experiments you run)
#
# The default experiments use DS02 and DS03:
#   mkdir -p $DATA_DIR/N-CMAPSS
#   # then copy N-CMAPSS_DS02.h5 and N-CMAPSS_DS03.h5 into that directory
#
# MZVAV (Multi-Zone VAV building HVAC dataset)
# --------------------------------------------
# MZVAV is not available on a public CDN. The dataset is from the ASHRAE
# Research Project 1312-RP (Multi-Zone VAV air handling unit). Contact
# the dataset authors or your institution's ASHRAE resource for access.
#
# Once obtained, place the CSV file at:
#   $DATA_DIR/building/MZVAV-2-2.csv
#
#   mkdir -p $DATA_DIR/building
#   cp /path/to/MZVAV-2-2.csv $DATA_DIR/building/

echo "Note: N-CMAPSS (concepts) and MZVAV require manual download."
echo "See comments in this file or docs/getting-started/datasets.md for instructions."
