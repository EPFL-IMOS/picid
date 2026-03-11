#!/usr/bin/env bash
# Run the picid_report pipeline and serve the report via a local HTTP server.
# Usage: ./picid_report/run.sh [port]
# Default port: 8000. Open http://localhost:8000/report_output/report.html after starting.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-report_output}"
PORT="${1:-8000}"

echo "Running pipeline -> $OUTPUT_DIR ..."
uv run python -m picid_report.run -o "$OUTPUT_DIR" --export-latex -q

echo ""
echo "Starting HTTP server at http://localhost:$PORT"
echo "Open: http://localhost:$PORT/report_output/report.html"
echo "Press Ctrl+C to stop."
exec uv run python -m http.server "$PORT"
