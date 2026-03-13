#!/bin/bash
#
# Master script: run all *_reproduce.sh in this directory.
# Intended to be submitted as a single job (e.g. sbatch, or bash run_all_reproduce.sh).
# Each reproduce script is run from the project root, one after another.
# Set STOP_ON_FAIL=1 to exit on first script failure; default is to continue.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root: parent of scripts/
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"

STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
count=0
failed=()

for script in "$SCRIPT_DIR"/*_reproduce.sh; do
  [[ -f "$script" ]] || continue
  name="$(basename "$script")"
  echo "=========================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $name ($((++count))/...) "
  echo "=========================================="
  if bash "$script"; then
    echo "[OK] $name"
  else
    echo "[FAIL] $name (exit $?)"
    failed+=("$name")
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      echo "STOP_ON_FAIL=1: exiting."
      exit 1
    fi
  fi
done

echo ""
echo "=========================================="
echo "Finished: $count script(s) run."
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed: ${failed[*]}"
  exit 1
fi
echo "All passed."
exit 0
