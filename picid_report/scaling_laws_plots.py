"""CLI for generating scaling law plots from W&B experiment data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent.parent))

from picid_report.core import load_runs_df, analyze_results
from picid_report.report import create_summary_table

console = Console()

REPORTING_METRICS = [
    "mse",
    "mae",
    "rmse",
    "mse_normalized",
    "mae_normalized",
    "rmse_normalized",
    "mse_denormalized",
    "mae_denormalized",
    "rmse_denormalized",
]
REPORTING_METRICS_BATTERY = [f"{r}_mean" for r in REPORTING_METRICS] + ["phme_score_mean"]

MODEL_RENAME = {
    "picid.model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper": "TabDPT",
    "picid.model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper": "TabPFN",
    "picid.model.forecasters.lstm_model.LSTM_Forecaster": "LSTM",
    "picid.model.estimators.cnn1d.wrapper.CNN1D_Wrapper": "1D-CNN",
    "picid.model.estimators.mlp.wrapper.MLPWrapper": "MLP",
    "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper": "StatBL",
    "picid.model.forecasters.spacetimeformer_model.Spacetimeformer_Forecaster": "STF",
    "picid.model.forecasters.crossformer_model.Crossformer_Forecaster": "CF",
    "picid.model.forecasters.timeseries_transformer_model.Timeseries_Transformer_Forecaster": "TST",
    "picid.model.forecasters.tide_model.TiDE_Forecaster": "TiDE",
    "picid.model.forecasters.patchtst_model.PatchTST_Forecaster": "PatchTST",
}

_NORMALIZED_VARS = [
    "test/mae_normalized",
    "test/rmse_normalized",
    "test/mae_denormalized",
    "test/rmse_denormalized",
]

DATASET_PLOT_CONFIGS: dict[str, dict] = {
    "unibo": {
        "metric": "test/mae_normalized",
        "y_label": "MAE (normalized)",
        "dataset_name": "Unibo",
        "value_vars": _NORMALIZED_VARS,
        "optimization_col": "test/mae_mean",
        "reporting_metrics": REPORTING_METRICS_BATTERY,
    },
    "n_cmapss_ds02": {
        "metric": "test/mae_normalized",
        "y_label": "MAE (normalized)",
        "dataset_name": "N-CMAPSS_DS02",
        "value_vars": _NORMALIZED_VARS,
        "optimization_col": "test/mae_denormalized",
        "reporting_metrics": REPORTING_METRICS,
    },
    "n_cmapss": {
        "metric": "test/rmse_denormalized",
        "y_label": "RMSE (denormalized)",
        "dataset_name": "N-CMAPSS",
        "value_vars": _NORMALIZED_VARS,
        "optimization_col": "test/mae_denormalized",
        "reporting_metrics": REPORTING_METRICS,
    },
    "phme20": {
        "metric": "test/mae_normalized",
        "y_label": "MAE (normalized)",
        "dataset_name": "PHME20",
        "value_vars": _NORMALIZED_VARS,
        "optimization_col": "test/mae_normalized",
        "reporting_metrics": REPORTING_METRICS,
    },
    "mzvav": {
        "metric": "test/f1",
        "y_label": "F1 Score",
        "dataset_name": "MZVAV",
        "value_vars": ["test/f1"],
        "optimization_col": "test/mae_normalized",
        "reporting_metrics": REPORTING_METRICS,
    },
}

_TABDPT_CLASS = "picid.model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper"

ALL_PROJECTS = [
    "n_cmapss_scaling",
    "16_02_2026_scaling_phme20_prognostics_raw",
    "16_02_2026_scaling_blockwise_phme20_prognostics_raw",
    "16_04_2026_scaling_mzvav_diagnostics_",
    "16_04_2026_scaling_blockwise_unibo_prognostics_combined",
    "16_04_2026_scaling_concepts_n_cmapss_ds02_prognostics_",
    "16_04_2026_scaling_blockwise_concepts_n_cmapss_ds02_prognostics_",
    "16_04_2026_scaling_unibo_prognostics_combined",
    "16_04_2026_scaling_blockwise_mzvav_diagnostics_",
]


def detect_dataset_type(project_name: str) -> str:
    if "nb14" in project_name or "unibo" in project_name:
        return "unibo"
    if "n_cmapss_ds02" in project_name:
        return "n_cmapss_ds02"
    if "n_cmapss" in project_name:
        return "n_cmapss"
    if "phme20" in project_name:
        return "phme20"
    if "mzvav" in project_name:
        return "mzvav"
    raise ValueError(
        f"Cannot detect dataset type for project '{project_name}'. "
        f"Expected one of: unibo/nb14, n_cmapss_ds02, n_cmapss, phme20, mzvav."
    )


def prepare_df_long(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    cfg = DATASET_PLOT_CONFIGS[dataset_type]
    value_vars: list[str] = cfg["value_vars"]
    id_vars = [
        "model._target_",
        "seed",
        "task_definition.subset_ratio",
        "task_definition.seq_len",
    ]

    df = df.copy()

    # Merge legacy column names into normalized ones when both exist
    if dataset_type in ("n_cmapss", "n_cmapss_ds02"):
        if "test/mae" in df.columns:
            for norm, legacy in [
                ("test/mae_normalized", "test/mae"),
                ("test/rmse_normalized", "test/rmse"),
                ("test/mae_denormalized", "test/mae"),
                ("test/rmse_denormalized", "test/rmse"),
            ]:
                if norm in df.columns:
                    df[norm] = df[norm].combine_first(df[legacy])

    # For multi-source n_cmapss, attach source label and keep only second source
    extra_id_vars: list[str] = []
    multisource_col = "datasource.multisource_data_splitter.sources_train"
    if dataset_type == "n_cmapss" and multisource_col in df.columns:
        df["ds"] = df[multisource_col].astype(str)
        extra_id_vars = ["ds"]

    available_vars = [v for v in value_vars if v in df.columns]
    keep_cols = id_vars + extra_id_vars + available_vars
    df_long = df[keep_cols].melt(
        id_vars=id_vars + extra_id_vars,
        value_vars=available_vars,
        var_name="metric",
        value_name="value",
    )

    df_long = df_long[
        ~((df_long["model._target_"] == _TABDPT_CLASS) & (df_long["task_definition.seq_len"] == 5))
    ]

    return df_long


def generate_plot(df_long: pd.DataFrame, dataset_type: str, output_dir: str) -> Path:
    cfg = DATASET_PLOT_CONFIGS[dataset_type]
    metric: str = cfg["metric"]
    y_label: str = cfg["y_label"]
    dataset_name: str = cfg["dataset_name"]

    plot_df = df_long
    if dataset_type == "n_cmapss" and "ds" in df_long.columns:
        unique_sources = df_long["ds"].unique()
        if len(unique_sources) > 1:
            plot_df = df_long[df_long["ds"] == unique_sources[1]]

    with sns.plotting_context("paper", font_scale=2, rc={"font.family": "Helvetica"}):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

        sns.lineplot(
            plot_df[
                plot_df["model._target_"].isin(MODEL_RENAME.keys())
                & (plot_df["metric"] == metric)
            ],
            x="task_definition.subset_ratio",
            y="value",
            hue="model._target_",
            hue_order=list(MODEL_RENAME.keys()),
            errorbar="sd",
            ax=ax,
        )

        ax.set_title(f"Performance vs. Randomly Selected Training Data Size [Dataset: {dataset_name}]")
        ax.set_xlabel("Fraction of the Data")
        ax.set_ylabel(y_label)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_axisbelow(True)
        ax.set_xlim(0.0, 1.0)
        ax.grid()

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

        ax.tick_params(axis="x", width=1.5)
        ax.tick_params(axis="y", width=1.5)

        old_legend = ax.legend_
        handles = old_legend.legend_handles
        labels = [MODEL_RENAME.get(t.get_text(), t.get_text()) for t in old_legend.get_texts()]
        old_legend.remove()

        legend = fig.legend(
            handles=handles,
            labels=labels,
            title="",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
            framealpha=1,
        )

        fig.tight_layout()

        output_path = Path(output_dir) / f"scaling_laws_{dataset_name}_{metric.split('/')[1]}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            transparent=True,
            bbox_extra_artists=(legend,),
            bbox_inches="tight",
        )
        plt.close(fig)

    return output_path


def display_summary_table(all_results: dict) -> None:
    summary_df = create_summary_table(all_results, precision=3)
    summary_df.index = summary_df.index.str.split(".").str[-1].str.replace("_", "-")
    console.print(Panel(summary_df.to_string(), title="Performance Summary", expand=False))


def run_project(project: str, user: str, output_dir: str, no_summary: bool) -> Path:
    """Run the full pipeline for a single W&B project and return the saved plot path."""
    console.print(
        Panel(
            f"[bold]Project:[/bold] {project}\n"
            f"[bold]User:[/bold]    {user}\n"
            f"[bold]Output:[/bold]  {output_dir}",
            title="Scaling Laws Plot",
            expand=False,
        )
    )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        t = progress.add_task("Loading W&B runs...", total=None)
        df, config_columns, dropped_columns = load_runs_df(project_name=project, user=user)
        progress.update(t, description="[green]W&B runs loaded.")

    dataset_type = detect_dataset_type(project)
    cfg = DATASET_PLOT_CONFIGS[dataset_type]
    console.print(
        f"Detected dataset type: [bold]{dataset_type}[/bold] → optimizing on [cyan]{cfg['optimization_col']}[/cyan]"
    )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        t = progress.add_task("Analysing results...", total=None)
        all_results = analyze_results(
            df=df,
            config_columns=config_columns,
            dropped_columns=dropped_columns,
            reporting_metrics=cfg["reporting_metrics"],
            metric_prefixes=["test/"],
            optimization_col=cfg["optimization_col"],
            optimization_mode="min",
        )
        progress.update(t, description="[green]Analysis complete.")

    if not no_summary:
        display_summary_table(all_results)

    df_long = prepare_df_long(df, dataset_type)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        t = progress.add_task("Generating plot...", total=None)
        output_path = generate_plot(df_long, dataset_type, output_dir)
        progress.update(t, description="[green]Plot generated.")

    console.print(Panel(f"[green]Saved:[/green] {output_path}", expand=False))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate scaling law plots from W&B experiment data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scaling_laws_plots.py -p 16_04_2026_scaling_blockwise_mzvav_diagnostics_\n"
            "  python scaling_laws_plots.py -p n_cmapss_scaling -u my-wandb-user -o ./out\n"
            "  python scaling_laws_plots.py --all\n"
        ),
    )
    parser.add_argument("--project", "-p", default=None, help="W&B project name")
    parser.add_argument(
        "--user", "-u", default="imos-buildingenergy-1", help="W&B username (default: imos-buildingenergy-1)"
    )
    parser.add_argument(
        "--output-dir", "-o", default="picid_report_artifacts", help="Output directory for PDFs (default: picid_report_artifacts)"
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip printing the summary table")
    parser.add_argument(
        "--all", dest="run_all", action="store_true",
        help=f"Run for all {len(ALL_PROJECTS)} hardcoded projects; overrides --project",
    )
    args = parser.parse_args()

    if not args.run_all and args.project is None:
        parser.error("one of --project / --all is required")

    projects = ALL_PROJECTS if args.run_all else [args.project]

    if args.run_all:
        console.print(
            Panel(
                "\n".join(f"  {p}" for p in projects),
                title=f"Running all {len(projects)} projects",
                expand=False,
            )
        )

    failed: list[tuple[str, str]] = []
    for project in projects:
        try:
            run_project(project, args.user, args.output_dir, args.no_summary)
        except Exception as exc:
            console.print(f"[red]FAILED {project}: {exc}[/red]")
            failed.append((project, str(exc)))

    if failed:
        console.print(Panel(
            "\n".join(f"[red]{p}[/red]: {e}" for p, e in failed),
            title=f"[red]{len(failed)} project(s) failed[/red]",
            expand=False,
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
