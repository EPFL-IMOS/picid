#!/usr/bin/env bash
# Phase 0 refactor: run characterization tests and report coverage for refactor target files.
# Target: ≥95% coverage on datasources base, data_objects, cache, preprocessing (preprocessor, base).
#
# (b) Utils run in isolation first (no data.py/awkward); then the rest.
# Some envs hit ArrowKeyError (pyarrow/awkward) when running with --cov; use --no-cov
# for a single full pass to confirm all 126 tests pass.
#
# Usage:
#   ./scripts/run_phase0_coverage.sh
#   ./scripts/run_phase0_coverage.sh --no-cov   # run tests only (single pass)
set -e
cd "$(dirname "$0")/.."

COV_ARGS=(
  --cov=picid.data.datasources.base
  --cov=picid.data.data_objects
  --cov=picid.data.cache.offline
  --cov=picid.data.preprocessing.preprocessor
  --cov=picid.data.preprocessing.base
  --cov-report=term-missing
  --cov-fail-under=0
)

# (a) Utils in isolation: only imports utils (no awkward)
UTILS_ONLY=(
  test/data/data_objects/test_utils_characterization.py
)

# (b) Rest of Phase 0 tests (loaders, data_objects, preprocessing, cache)
PHASE0_FILES=(
  test/data/datasources/base/test_loaders_characterization.py
  test/data/datasources/base/test_phmd_loader_characterization.py
  test/data/data_objects/test_data_objects_characterization.py
  test/data/preprocessing/test_pipeline_characterization.py
  test/data/preprocessing/test_preprocessor.py
  test/data/preprocessing/test_preprocessor_caching.py
  test/data/cache/test_file_system_cache.py
)

if [[ " ${*} " =~ " --no-cov " ]]; then
  echo "=== Phase 0 tests (single run) ==="
  uv run pytest "${UTILS_ONLY[@]}" "${PHASE0_FILES[@]}" -q "${@}"
else
  echo "=== Step 1: utils in isolation ==="
  uv run pytest "${UTILS_ONLY[@]}" --cov=picid.data.data_objects.utils --cov-report=term-missing -q
  echo "=== Step 2: rest of Phase 0 with coverage ==="
  uv run pytest "${PHASE0_FILES[@]}" "${COV_ARGS[@]}" -q "$@"
fi
