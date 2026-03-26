#!/usr/bin/env bash
# Generate HTML reports for every project listed in projects.sh.
# Each report is written to report_output/<PROJECT_NAME>/<PROJECT_NAME>.html
# (report filename is exactly the project name with .html).
# Usage: ./picid_report/run_all_projects.sh [output_base_dir] [-- extra_args...]
# Default output base: report_output (same as run.sh).
# Pass --data-first after -- to disable legacy search space fallback (data-first mode when search_space.py has no entry).
# Example: ./picid_report/run_all_projects.sh report_output -- --data-first

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PROJECTS_FILE="${SCRIPT_DIR}/projects.sh"
OUTPUT_BASE="report_output"
EXTRA_ARGS=()
if [[ "$1" == "--" ]]; then
  shift
  while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done
elif [[ -n "$1" && "$2" == "--" ]]; then
  OUTPUT_BASE="$1"
  shift 2
  while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done
elif [[ -n "$1" ]]; then
  OUTPUT_BASE="$1"
  shift
fi

if [[ ! -f "$PROJECTS_FILE" ]]; then
  echo "Error: projects file not found: $PROJECTS_FILE" >&2
  exit 1
fi

while IFS= read -r proj || [[ -n "$proj" ]]; do
  proj="${proj#"${proj%%[![:space:]]*}"}"
  proj="${proj%"${proj##*[![:space:]]}"}"
  [[ -z "$proj" ]] && continue
  [[ "$proj" =~ ^# ]] && continue
  echo "=============================================="
  echo "Project: $proj"
  echo "=============================================="
  uv run python -m picid_report.run \
    --project "$proj" \
    -o "$OUTPUT_BASE/$proj" \
    --report-name "${proj}.html" \
    --export-latex \
    "${EXTRA_ARGS[@]}"
  echo ""
done < "$PROJECTS_FILE"


echo "=============================================="
echo "Generating average rank plots..."
echo "=============================================="
uv run python -m picid_report.scripts.plot_average_rank --output-dir "$OUTPUT_BASE" "${EXTRA_ARGS[@]}" || \
  echo "Warning: plot_average_rank failed (non-fatal)" >&2



echo "=============================================="
echo "Generating cross-dataset summary heatmaps..."
echo "=============================================="
uv run python -m picid_report.scripts.plot_summary_all -o "$OUTPUT_BASE" "${EXTRA_ARGS[@]}" || \
  echo "Warning: plot_summary_all failed (non-fatal)" >&2

echo "Done. Reports under $OUTPUT_BASE/<project_name>/<project_name>.html"
echo "Cross-dataset heatmaps: $OUTPUT_BASE/plots/summary_regression.pdf, summary_classification.pdf"
