#!/usr/bin/env bash
# Run nox experiment sessions (diagnostics + prognostics) on a server.
# Output is tee'd to logs/nox_<timestamp>.log for inspection after the run.
set -euo pipefail
cd "$(dirname "$0")/.."

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nox_${TIMESTAMP}.log"

echo "=== nox run: experiments_diagnostics + experiments_prognostics ===" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE"

uv run nox -s experiments_diagnostics 2>&1 | tee -a "$LOG_FILE"
uv run nox -s experiments_prognostics 2>&1 | tee -a "$LOG_FILE"

echo "=== Done. Log saved to $LOG_FILE ==="
