"""CLI for generating missing-data ablation bar charts from W&B experiment data."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

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

METRIC = "test/mae_normalized"
Y_LABEL = "MAE (normalized)"
DATASET_NAME = "PHME20"
OPTIMIZATION_COL = "test/mae_normalized"

PROJECT = "17_02_2026_missing_data_phme20_prognostics_ablation_missing_values"

# Detect the NaN-variant TabPFN by looking for "tabpfn_fit_predict_nan" in run_name.
# Both variants share the same _target_ class — the suffix distinguishes them here.
_NAN_SUFFIX = " (nan)"

MODEL_RENAME = {
    # TabPFN NaN variant (must come before plain TabPFN key)
    "lmetk.model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper (nan)": "TabPFN+NaN",
    # current picid.model.* paths
    "picid.model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper": "TabDPT",
    "picid.model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper": "TabPFN",
    "picid.model.forecasters.lstm_model.LSTM_Forecaster": "LSTM",
    "picid.model.estimators.cnn1d.wrapper.CNN1D_Wrapper": "1D-CNN",
    "picid.model.estimators.mlp.wrapper.MLPWrapper": "MLP",
    "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper (linear)": "StatBL-lin",
    "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper (exponential)": "StatBL-exp",
    "picid.model.forecasters.spacetimeformer_model.Spacetimeformer_Forecaster": "STF",
    "picid.model.forecasters.crossformer_model.Crossformer_Forecaster": "CF",
    "picid.model.forecasters.timeseries_transformer_model.Timeseries_Transformer_Forecaster": "TST",
    "picid.model.forecasters.tide_model.TiDE_Forecaster": "TiDE",
    "picid.model.forecasters.patchtst_model.PatchTST_Forecaster": "PatchTST",
    # intermediate picid.model.wrappers.* paths
    "picid.model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": "1D-CNN",
    "picid.model.wrappers.mlp_wrapper.MLPWrapper": "MLP",
    "picid.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": "StatBL-lin",
    "picid.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": "StatBL-exp",
    "picid.baselines.lstm_model.LSTM_Forecaster": "LSTM",
    "picid.baselines.spacetimeformer_model.Spacetimeformer_Forecaster": "STF",
    "picid.baselines.crossformer_model.Crossformer_Forecaster": "CF",
    "picid.baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": "TST",
    "picid.baselines.tide_model.TiDE_Forecaster": "TiDE",
    "picid.baselines.patchtst_model.PatchTST_Forecaster": "PatchTST",
    # picid.model.estimators.* paths
    "picid.model.estimators.tabdpt.wrapper.FitPredictTabDPTWrapper": "TabDPT",
    "picid.model.estimators.tabpfn.wrapper.FitPredictTabPFNWrapper": "TabPFN",
    # legacy lmetk.* paths
    "lmetk.model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper": "TabDPT",
    "lmetk.model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper": "TabPFN",
    "lmetk.baselines.lstm_model.LSTM_Forecaster": "LSTM",
    "lmetk.model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": "1D-CNN",
    "lmetk.model.wrappers.mlp_wrapper.MLPWrapper": "MLP",
    "lmetk.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": "StatBL-lin",
    "lmetk.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": "StatBL-exp",
    "lmetk.baselines.spacetimeformer_model.Spacetimeformer_Forecaster": "STF",
    "lmetk.baselines.crossformer_model.Crossformer_Forecaster": "CF",
    "lmetk.baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": "TST",
    "lmetk.baselines.tide_model.TiDE_Forecaster": "TiDE",
    "lmetk.baselines.patchtst_model.PatchTST_Forecaster": "PatchTST",
}

FM_LABELS = {"TabDPT", "TabPFN", "TabPFN+NaN"}

LEGEND_ORDER = [
    "TabDPT", "TabPFN", "TabPFN+NaN",
    "LSTM", "1D-CNN", "MLP",
    "StatBL-lin", "StatBL-exp",
    "STF", "CF", "TST", "TiDE", "PatchTST",
]

MODEL_COLORS = {
    "TabDPT":     "#e41a1c",
    "TabPFN":     "#ff7f00",
    "TabPFN+NaN": "#fdbf6f",
    "LSTM":       "#377eb8",
    "1D-CNN":     "#4daf4a",
    "MLP":        "#984ea3",
    "StatBL":     "#a65628",
    "StatBL-lin": "#a65628",
    "StatBL-exp": "#f781bf",
    "STF":        "#999999",
    "CF":         "#66c2a5",
    "TST":        "#fc8d62",
    "TiDE":       "#8da0cb",
    "PatchTST":   "#a6d854",
}


def _coerce_list_cells(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Tag TabPFN NaN variant and StatBL model types, then return a clean copy."""
    df = df.copy()

    # Tag TabPFN NaN variant: append _NAN_SUFFIX to model._target_ for those runs.
    # Detect via run_name which contains "tabpfn_fit_predict_nan".
    tabpfn_nan_mask = df["run_name"].str.contains("tabpfn_fit_predict_nan", na=False)
    df.loc[tabpfn_nan_mask, "model._target_"] = (
        df.loc[tabpfn_nan_mask, "model._target_"] + _NAN_SUFFIX
    )

    # Tag StatBL model type (linear / exponential) — mirrors scaling_laws_plots.py
    model_type_col = "model.model_type"
    if model_type_col in df.columns:
        mask = df[model_type_col].notna()
        df.loc[mask, "model._target_"] = (
            df.loc[mask, "model._target_"]
            + " ("
            + df.loc[mask, model_type_col]
            + ")"
        )

    return df


def compute_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model mean ± std for the primary metric, sorted best→worst."""
    present = df[METRIC].notna()
    stats = (
        df.loc[present]
        .groupby("model._target_")[METRIC]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .rename(columns={"model._target_": "model_key"})
    )
    stats["std"] = stats["std"].fillna(0.0)
    stats["label"] = stats["model_key"].map(
        lambda k: MODEL_RENAME.get(k, k.split(".")[-1])
    )
    # Sort by LEGEND_ORDER, then alphabetically for any unknowns
    def _order(label: str) -> int:
        return LEGEND_ORDER.index(label) if label in LEGEND_ORDER else len(LEGEND_ORDER)

    stats["_order"] = stats["label"].map(_order)
    stats = stats.sort_values("mean").reset_index(drop=True)
    return stats


def display_stats_table(stats: pd.DataFrame) -> None:
    table = Table(
        title=f"Model performance  (metric: {METRIC}, dataset: {DATASET_NAME})",
        show_lines=True,
    )
    table.add_column("Model", style="bold", no_wrap=True)
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("N", justify="right")

    for _, row in stats.iterrows():
        label = row["label"]
        style = "bold" if label in FM_LABELS else ""
        table.add_row(
            f"[{style}]{label}[/{style}]" if style else label,
            f"{row['mean']:.4f}",
            f"{row['std']:.4f}",
            str(int(row["count"])),
        )
    console.print(table)


def display_summary_table(all_results: dict) -> None:
    summary_df = create_summary_table(all_results, precision=3)
    summary_df.index = summary_df.index.str.split(".").str[-1].str.replace("_", "-")
    console.print(
        Panel(summary_df.to_string(), title="Performance Summary", expand=False)
    )


def generate_plot(
    stats: pd.DataFrame,
    output_dir: str,
    project_name: str = "",
    no_title: bool = False,
) -> Path:
    labels = stats["label"].tolist()
    means = stats["mean"].to_numpy()
    stds = stats["std"].to_numpy()
    colors = [MODEL_COLORS.get(lbl, "#333333") for lbl in labels]

    x = np.arange(len(labels))

    with sns.plotting_context("paper", font_scale=2, rc={"font.family": "Helvetica"}):
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 7))

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            capsize=5,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
        )

        # Bold edge for FM models
        for bar, lbl in zip(bars, labels):
            if lbl in FM_LABELS:
                bar.set_linewidth(2.0)
                bar.set_edgecolor("black")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"$\\bf{{{lbl}}}$" if lbl in FM_LABELS else lbl
                for lbl in labels
            ],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel(Y_LABEL)
        ax.set_xlabel("Model")

        if not no_title:
            ax.set_title(
                f"Performance on Missing-Data Ablation [Dataset: {DATASET_NAME}]"
            )

        ax.set_axisbelow(True)
        ax.grid(axis="y")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

        ax.tick_params(axis="x", width=1.5)
        ax.tick_params(axis="y", width=1.5)

        fig.tight_layout()

        date_match = re.search(r"\d{2}_\d{2}_\d{4}", project_name)
        date_prefix = f"{date_match.group()}_" if date_match else ""
        output_path = (
            Path(output_dir)
            / f"{date_prefix}missing_data_{DATASET_NAME}_{METRIC.split('/')[1]}.pdf"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)

    return output_path


def run_project(
    project: str,
    user: str,
    output_dir: str,
    no_summary: bool,
    no_title: bool = False,
) -> Path:
    console.print(
        Panel(
            f"[bold]Project:[/bold] {project}\n"
            f"[bold]User:[/bold]    {user}\n"
            f"[bold]Output:[/bold]  {output_dir}",
            title="Missing-Data Ablation Plot",
            expand=False,
        )
    )

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        t = progress.add_task("Loading W&B runs...", total=None)
        df, config_columns, dropped_columns = load_runs_df(
            project_name=project, user=user
        )
        progress.update(t, description="[green]W&B runs loaded.")

    df = _coerce_list_cells(df)
    df = prepare_df(df)

    if not no_summary:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            t = progress.add_task("Analysing results...", total=None)
            all_results = analyze_results(
                df=df,
                config_columns=config_columns,
                dropped_columns=dropped_columns,
                reporting_metrics=REPORTING_METRICS,
                metric_prefixes=["test/"],
                optimization_col=OPTIMIZATION_COL,
                optimization_mode="min",
            )
            progress.update(t, description="[green]Analysis complete.")
        display_summary_table(all_results)

    stats = compute_model_stats(df)
    display_stats_table(stats)

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        t = progress.add_task("Generating plot...", total=None)
        output_path = generate_plot(stats, output_dir, project, no_title=no_title)
        progress.update(t, description="[green]Plot generated.")

    console.print(Panel(f"[green]Saved:[/green] {output_path}", expand=False))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate missing-data ablation bar charts from W&B experiment data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            f"  python missing_data_plots.py\n"
            f"  python missing_data_plots.py -p {PROJECT}\n"
            "  python missing_data_plots.py -u my-wandb-user -o ./out --no-title\n"
        ),
    )
    parser.add_argument(
        "--project",
        "-p",
        default=PROJECT,
        help=f"W&B project name (default: {PROJECT})",
    )
    parser.add_argument(
        "--user",
        "-u",
        default="imos-buildingenergy-1",
        help="W&B username (default: imos-buildingenergy-1)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="picid_report_artifacts",
        help="Output directory for PDFs (default: picid_report_artifacts)",
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="Skip printing the summary table"
    )
    parser.add_argument(
        "--no-title", action="store_true", help="Omit plot title (for publication figures)"
    )
    args = parser.parse_args()

    run_project(
        args.project,
        args.user,
        args.output_dir,
        args.no_summary,
        no_title=args.no_title,
    )


if __name__ == "__main__":
    main()
