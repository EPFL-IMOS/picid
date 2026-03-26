"""
Cross-dataset summary heatmap: loads all results.nc files and calls plot_summary.

Scans <output_base>/*/results.nc, combines them via combine_results_nc, then
produces two heatmaps saved under <output_base>/plots/:
  - summary_regression.pdf/png   (metric: test_best_rerun/mae_normalized, mode: min)
  - summary_classification.pdf/png (metric: test_best_rerun/accuracy, mode: max)

If either metric is absent in the combined dataset the corresponding plot is skipped.

Usage:
  python -m picid_report.scripts.plot_summary_all --output-dir report_output
  python -m picid_report.scripts.plot_summary_all -o report_output --regression-metric test_best_rerun/mae_normalized --classification-metric test_best_rerun/accuracy
"""

import argparse
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_REGRESSION_METRIC = "test_best_rerun/mae_normalized"
DEFAULT_CLASSIFICATION_METRIC = "test_best_rerun/accuracy"


def run(
    output_base: str,
    regression_metric: str = DEFAULT_REGRESSION_METRIC,
    classification_metric: str = DEFAULT_CLASSIFICATION_METRIC,
) -> None:
    from picid_report.run import combine_results_nc
    from picid_report.report.plots import plot_summary

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting requires matplotlib: pip install matplotlib") from e

    ds = combine_results_nc(output_base)
    available = list(ds.coords["metric_key"].values)
    logger.info("Combined dataset: %d datasets, %d models, %d metric_keys",
                ds.sizes["dataset"], ds.sizes["model"], ds.sizes["metric_key"])

    plots_dir = os.path.join(output_base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def _fallback(metric: str):
        """Strip _best_rerun so fit-predict models fill from their plain test/* metric."""
        stripped = metric.replace("_best_rerun", "") if "_best_rerun" in metric else None
        return [stripped] if stripped and stripped != metric else None

    tasks = [
        (regression_metric, "min", _fallback(regression_metric), "summary_regression",
         "Regression overview — " + regression_metric),
        (classification_metric, "max", _fallback(classification_metric), "summary_classification",
         "Classification overview — " + classification_metric),
    ]

    for metric_key, mode, fallbacks, filename, title in tasks:
        if metric_key not in available:
            logger.warning("Metric %r not found in combined dataset — skipping %s", metric_key, filename)
            continue
        save_path = os.path.join(plots_dir, filename + ".png")
        fig = plot_summary(ds, metric_key=metric_key, mode=mode, fallback_metric_keys=fallbacks,
                           save_path=save_path, title=title)
        if fig is not None:
            plt.close(fig)
            logger.info("Saved %s.pdf/.png", os.path.join(plots_dir, filename))
        else:
            logger.warning("plot_summary returned None for %r", metric_key)

    ds.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate cross-dataset summary heatmaps from results.nc files."
    )
    parser.add_argument("--output-dir", "-o", default="report_output",
                        help="Base directory containing per-project report folders (default: report_output)")
    parser.add_argument("--regression-metric", default=DEFAULT_REGRESSION_METRIC,
                        help=f"metric_key for regression heatmap (default: {DEFAULT_REGRESSION_METRIC})")
    parser.add_argument("--classification-metric", default=DEFAULT_CLASSIFICATION_METRIC,
                        help=f"metric_key for classification heatmap (default: {DEFAULT_CLASSIFICATION_METRIC})")
    args = parser.parse_args()

    run(
        output_base=args.output_dir,
        regression_metric=args.regression_metric,
        classification_metric=args.classification_metric,
    )


if __name__ == "__main__":
    main()
