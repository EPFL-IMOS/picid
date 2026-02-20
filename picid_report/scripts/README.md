# Scripts

## plot_average_rank.py

Produces bar plots of **average rank** of each model across datasets (from existing `summary.csv` files under report output).

**Two modes:**

- **Task-type mode (default):** Produces two plot sets — one for **classification** datasets (metrics like accuracy, f1) and one for **regression** (loss, mse, mae, etc.). Each CSV is classified by the metrics in its header; ranking uses test metrics with preference for `test_best_rerun/*` and fallback to `test/*` (fit_predict). For regression, normalized variants are preferred over denormalized. Outputs: `average_rank_classification.png` and `average_rank_regression.png` (and optional CSVs).
- **Single-metric mode:** Pass `--rank-metric <name>` to use one metric for all files and produce a single plot (e.g. `val_best_rerun/loss` or `test/loss`).

**Usage (after running the report pipeline):**

```bash
# Default: task-type mode, writes report_output/plots/average_rank_classification.png and average_rank_regression.png
python -m picid_report.scripts.plot_average_rank --output-dir report_output

# With CSV export
python -m picid_report.scripts.plot_average_rank -o report_output --csv report_output/plots

# Only regression plot
python -m picid_report.scripts.plot_average_rank -o report_output --task-type regression

# Single-metric mode (one plot)
python -m picid_report.scripts.plot_average_rank -o report_output --rank-metric val_best_rerun/loss --out report_output/plots/average_rank.png --csv report_output/plots/average_rank.csv
```

**Options:** `--output-dir`, `--rank-metric` (optional; if set, single-metric mode), `--task-type` (both | classification | regression), `--rank-mode` (min/max, single-metric only), `--out` (directory in task-type mode, file path in single-metric mode), `--csv`.
