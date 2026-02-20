#!/usr/bin/env bash
# Run coverage over the whole caching stack (FileSystemCache + PreProcessor pipeline caching).
#
# Usage:
#   ./scripts/run_caching_coverage.sh           # full: cache package + preprocessor tests
#   ./scripts/run_caching_coverage.sh --cache-only   # cache package only (no awkward/pyarrow)
set -e
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
CACHE_ONLY=false
for arg in "$@"; do
  if [ "$arg" = "--cache-only" ]; then CACHE_ONLY=true; else EXTRA_ARGS+=("$arg"); fi
done

if [ "$CACHE_ONLY" = true ]; then
  echo "=== Caching coverage (cache package only) ==="
  uv run pytest \
    test/data/cache/test_file_system_cache.py \
    test/data/cache/test_stream_cache.py \
    test/data/cache/test_file_lock.py \
    --cov=picid.data.cache \
    --cov-report=term-missing \
    --cov-report=html:htmlcov_caching \
    -q \
    "${EXTRA_ARGS[@]}"
else
  echo "=== Caching coverage (cache package + preprocessor pipeline) ==="
  uv run pytest \
    test/data/cache/test_file_system_cache.py \
    test/data/cache/test_stream_cache.py \
    test/data/cache/test_file_lock.py \
    test/data/preprocessing/test_preprocessor_caching.py \
    --cov=picid.data.cache \
    --cov=picid.data.preprocessing.preprocessor \
    --cov-report=term-missing \
    --cov-report=html:htmlcov_caching \
    --cov-fail-under=0 \
    -q \
    "${EXTRA_ARGS[@]}"
fi
