#!/usr/bin/env bash
# Generate HTML reports for every project listed in projects.sh.
# Each report is written to report_output/<PROJECT_NAME>/<PROJECT_NAME>.html
# (report filename is exactly the project name with .html).
# Usage: ./lmetk_report/run_all_projects.sh [output_base_dir]
# Default output base: report_output (same as run.sh).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PROJECTS_FILE="${SCRIPT_DIR}/projects.sh"
OUTPUT_BASE="${1:-report_output}"

if [[ ! -f "$PROJECTS_FILE" ]]; then
  echo "Error: projects file not found: $PROJECTS_FILE" >&2
  exit 1
fi

while IFS= read -r proj || [[ -n "$proj" ]]; do
  proj="${proj#"${proj%%[![:space:]]*}"}"
  proj="${proj%"${proj##*[![:space:]]}"}"
  [[ -z "$proj" ]] && continue
  echo "=============================================="
  echo "Project: $proj"
  echo "=============================================="
  uv run python -m lmetk_report.run \
    --project "$proj" \
    -o "$OUTPUT_BASE/$proj" \
    --report-name "${proj}.html" \
    --export-latex
  echo ""
done < "$PROJECTS_FILE"

echo "Done. Reports under $OUTPUT_BASE/<project_name>/<project_name>.html"
