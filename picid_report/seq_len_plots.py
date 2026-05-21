"""CLI for generating sequence-length experiment plots from W&B experiment data."""

from __future__ import annotations

import argparse
import re
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
REPORTING_METRICS_BATTERY = [f"{r}_mean" for r in REPORTING_METRICS] + [
    "phme_score_mean"
]

MODEL_RENAME = {
    # current picid.model.* paths
    "picid.model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper": "TabDPT",
    "picid.model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper": "TabPFN",
    "picid.model.forecasters.lstm_model.LSTM_Forecaster": "LSTM",
    "picid.model.estimators.cnn1d.wrapper.CNN1D_Wrapper": "1D-CNN",
    "picid.model.estimators.mlp.wrapper.MLPWrapper": "MLP",
    "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper": "StatBL",
    "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper (linear)": "StatBL-lin",
    "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper (exponential)": "StatBL-exp",
    "picid.model.forecasters.spacetimeformer_model.Spacetimeformer_Forecaster": "STF",
    "picid.model.forecasters.crossformer_model.Crossformer_Forecaster": "CF",
    "picid.model.forecasters.timeseries_transformer_model.Timeseries_Transformer_Forecaster": "TST",
    "picid.model.forecasters.tide_model.TiDE_Forecaster": "TiDE",
    "picid.model.forecasters.patchtst_model.PatchTST_Forecaster": "PatchTST",
    # intermediate picid.baselines.* / picid.model.wrappers.* paths
    "picid.model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": "1D-CNN",
    "picid.model.wrappers.mlp_wrapper.MLPWrapper": "MLP",
    "picid.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper": "StatBL",
    "picid.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": "StatBL-lin",
    "picid.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": "StatBL-exp",
    "picid.baselines.lstm_model.LSTM_Forecaster": "LSTM",
    "picid.baselines.spacetimeformer_model.Spacetimeformer_Forecaster": "STF",
    "picid.baselines.crossformer_model.Crossformer_Forecaster": "CF",
    "picid.baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": "TST",
    "picid.baselines.tide_model.TiDE_Forecaster": "TiDE",
    "picid.baselines.patchtst_model.PatchTST_Forecaster": "PatchTST",
    # picid.model.estimators.* paths (another relocation)
    "picid.model.estimators.tabdpt.wrapper.FitPredictTabDPTWrapper": "TabDPT",
    "picid.model.estimators.tabpfn.wrapper.FitPredictTabPFNWrapper": "TabPFN",
    # legacy lmetk.* paths
    "lmetk.model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper": "TabDPT",
    "lmetk.model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper": "TabPFN",
    "lmetk.baselines.lstm_model.LSTM_Forecaster": "LSTM",
    "lmetk.model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": "1D-CNN",
    "lmetk.model.wrappers.mlp_wrapper.MLPWrapper": "MLP",
    "lmetk.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper": "StatBL",
    "lmetk.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": "StatBL-lin",
    "lmetk.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": "StatBL-exp",
    "lmetk.baselines.spacetimeformer_model.Spacetimeformer_Forecaster": "STF",
    "lmetk.baselines.crossformer_model.Crossformer_Forecaster": "CF",
    "lmetk.baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": "TST",
    "lmetk.baselines.tide_model.TiDE_Forecaster": "TiDE",
    "lmetk.baselines.patchtst_model.PatchTST_Forecaster": "PatchTST",
}

FM_LABELS = {"TabDPT", "TabPFN"}

LEGEND_ORDER = [
    "TabDPT", "TabPFN",
    "LSTM", "1D-CNN", "MLP",
    "StatBL", "StatBL-lin", "StatBL-exp",
    "STF", "CF", "TST", "TiDE", "PatchTST",
]

MODEL_COLORS = {
    "TabDPT":     "#e41a1c",
    "TabPFN":     "#ff7f00",
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

_NORMALIZED_VARS = [
    "test/mae_normalized",
    "test/rmse_normalized",
    "test/mae_denormalized",
    "test/rmse_denormalized",
]

DATASET_PLOT_CONFIGS: dict[str, dict] = {
    "xjtu_sy": {
        "metric": "test/mae_normalized",
        "y_label": "MAE (normalized)",
        "dataset_name": "XJTU-SY",
        "value_vars": _NORMALIZED_VARS,
        "optimization_col": "test/mae_mean",
        "reporting_metrics": REPORTING_METRICS_BATTERY,
    },
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
}

_TABDPT_CLASSES = {
    "picid.model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper",
    "picid.model.estimators.tabdpt.wrapper.FitPredictTabDPTWrapper",
    "lmetk.model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper",
}

ALL_PROJECTS = [
    "16_04_2026_seq_len_experiments_xjtu_sy_prognostics_combined",
    "16_04_2026_seq_len_experiments_concepts_n_cmapss_ds02_prognostics_",
    "16_04_2026_seq_len_experiments_phme20_prognostics_raw",
    "16_04_2026_seq_len_experiments_unibo_prognostics_combined",
]


def detect_dataset_type(project_name: str) -> str:
    if "xjtu_sy" in project_name:
        return "xjtu_sy"
    if "nb14" in project_name or "unibo" in project_name:
        return "unibo"
    if "n_cmapss_ds02" in project_name:
        return "n_cmapss_ds02"
    if "n_cmapss" in project_name:
        return "n_cmapss"
    if "phme20" in project_name:
        return "phme20"
    raise ValueError(
        f"Cannot detect dataset type for project '{project_name}'. "
        f"Expected one of: xjtu_sy, unibo/nb14, n_cmapss_ds02, n_cmapss, phme20."
    )


def _coerce_list_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Stringify list-valued cells so pandas groupby can hash them."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    return df


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

    extra_id_vars: list[str] = []
    multisource_col = "datasource.multisource_data_splitter.sources_train"
    if dataset_type == "n_cmapss" and multisource_col in df.columns:
        df["ds"] = df[multisource_col].astype(str)
        extra_id_vars.append("ds")

    model_type_col = "model.model_type"
    if model_type_col in df.columns:
        extra_id_vars.append(model_type_col)

    available_vars = [v for v in value_vars if v in df.columns]
    keep_cols = id_vars + extra_id_vars + available_vars
    df_long = df[keep_cols].melt(
        id_vars=id_vars + extra_id_vars,
        value_vars=available_vars,
        var_name="metric",
        value_name="value",
    )

    if model_type_col in df_long.columns:
        mask = df_long[model_type_col].notna()
        df_long.loc[mask, "model._target_"] = (
            df_long.loc[mask, "model._target_"]
            + " ("
            + df_long.loc[mask, model_type_col]
            + ")"
        )
        df_long = df_long.drop(columns=[model_type_col])

    df_long = df_long[
        ~(
            df_long["model._target_"].isin(_TABDPT_CLASSES)
            & (df_long["task_definition.seq_len"] == 5)
        )
    ]

    return df_long


def _parse_errorbar(value: str):
    """Parse CLI errorbar string into a value accepted by seaborn.

    Accepts:
      "sd", "se", "ci", "pi"        → passed as-is
      "pi,95" / "ci,95"             → ("pi", 95)
      "none"                        → None
    """
    if value.lower() == "none":
        return None
    if "," in value:
        kind, level = value.split(",", 1)
        return (kind.strip(), int(level.strip()))
    return value


def generate_plot(
    df_long: pd.DataFrame,
    dataset_type: str,
    output_dir: str,
    project_name: str = "",
    errorbar: str | tuple | None = "sd",
    no_title: bool = False,
) -> Path:
    cfg = DATASET_PLOT_CONFIGS[dataset_type]
    metric: str = cfg["metric"]
    y_label: str = cfg["y_label"]
    dataset_name: str = cfg["dataset_name"]

    plot_df = df_long
    if dataset_type == "n_cmapss" and "ds" in df_long.columns:
        unique_sources = df_long["ds"].unique()
        if len(unique_sources) > 1:
            plot_df = df_long[df_long["ds"] == unique_sources[1]]

    present_models = set(plot_df["model._target_"].unique())
    active_rename = {k: v for k, v in MODEL_RENAME.items() if k in present_models}
    if not active_rename:
        raise ValueError(
            f"No models in the data matched MODEL_RENAME for dataset '{dataset_type}'. "
            f"Models in data: {sorted(present_models)}"
        )

    active_rename = dict(
        sorted(
            active_rename.items(),
            key=lambda kv: LEGEND_ORDER.index(kv[1]) if kv[1] in LEGEND_ORDER else len(LEGEND_ORDER),
        )
    )
    palette = {k: MODEL_COLORS.get(v, "#333333") for k, v in active_rename.items()}

    with sns.plotting_context("paper", font_scale=2, rc={"font.family": "Helvetica"}):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

        sns.lineplot(
            plot_df[
                plot_df["model._target_"].isin(active_rename.keys())
                & (plot_df["metric"] == metric)
            ],
            x="task_definition.seq_len",
            y="value",
            hue="model._target_",
            hue_order=list(active_rename.keys()),
            palette=palette,
            errorbar=errorbar,
            ax=ax,
        )

        fm_keys = {k for k, v in active_rename.items() if v in FM_LABELS}
        for i, key in enumerate(active_rename.keys()):
            if key in fm_keys and i < len(ax.lines):
                lw = ax.lines[i].get_linewidth()
                ax.lines[i].set_linewidth(lw * 2)

        if not no_title:
            ax.set_title(
                f"Performance vs. Sequence Length [Dataset: {dataset_name}]"
            )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(y_label)
        ax.set_axisbelow(True)
        ax.grid()

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

        ax.tick_params(axis="x", width=1.5)
        ax.tick_params(axis="y", width=1.5)

        if ax.legend_ is None:
            raise ValueError(
                f"No legend was created; the metric '{metric}' may be missing from the filtered data."
            )
        old_legend = ax.legend_
        handles = old_legend.legend_handles
        labels = [
            active_rename.get(t.get_text(), t.get_text())
            for t in old_legend.get_texts()
        ]
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

        for text in legend.get_texts():
            if text.get_text() in FM_LABELS:
                text.set_fontweight("bold")

        fig.tight_layout()

        date_match = re.search(r"\d{2}_\d{2}_\d{4}", project_name)
        date_prefix = f"{date_match.group()}_" if date_match else ""
        output_path = (
            Path(output_dir)
            / f"{date_prefix}seq_len_{dataset_name}_{metric.split('/')[1]}.pdf"
        )
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


def display_run_counts_table(df_long: pd.DataFrame, dataset_type: str) -> None:
    metric = DATASET_PLOT_CONFIGS[dataset_type]["metric"]
    df_metric = df_long[df_long["metric"] == metric]

    counts = (
        df_metric.groupby(["model._target_", "task_definition.seq_len"])
        .size()
        .unstack("task_definition.seq_len")
        .sort_index(axis=1)
    )
    counts.index = counts.index.map(lambda x: MODEL_RENAME.get(x, x.split(".")[-1]))

    seq_len_cols = [str(int(c)) for c in counts.columns]
    table = Table(
        title=f"Runs per model × seq_len  (metric: {metric})", show_lines=True
    )
    table.add_column("Model", style="bold", no_wrap=True)
    for col in seq_len_cols:
        table.add_column(col, justify="right")

    for model_name, row in counts.iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append("[yellow]-[/yellow]")
            elif int(v) != 5:
                cells.append(f"[yellow]{int(v)}[/yellow]")
            else:
                cells.append(str(int(v)))
        table.add_row(model_name, *cells)

    console.print(table)


def display_summary_table(all_results: dict) -> None:
    summary_df = create_summary_table(all_results, precision=3)
    summary_df.index = summary_df.index.str.split(".").str[-1].str.replace("_", "-")
    console.print(
        Panel(summary_df.to_string(), title="Performance Summary", expand=False)
    )


def run_project(project: str, user: str, output_dir: str, no_summary: bool, errorbar: str | tuple | None = "sd", no_title: bool = False) -> Path:
    """Run the full pipeline for a single W&B project and return the saved plot path."""
    console.print(
        Panel(
            f"[bold]Project:[/bold] {project}\n"
            f"[bold]User:[/bold]    {user}\n"
            f"[bold]Output:[/bold]  {output_dir}",
            title="Sequence Length Plot",
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

    dataset_type = detect_dataset_type(project)
    cfg = DATASET_PLOT_CONFIGS[dataset_type]
    console.print(
        f"Detected dataset type: [bold]{dataset_type}[/bold] → optimizing on [cyan]{cfg['optimization_col']}[/cyan]"
    )

    df = _coerce_list_cells(df)

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
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
    display_run_counts_table(df_long, dataset_type)

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        t = progress.add_task("Generating plot...", total=None)
        output_path = generate_plot(df_long, dataset_type, output_dir, project, errorbar=errorbar, no_title=no_title)
        progress.update(t, description="[green]Plot generated.")

    console.print(Panel(f"[green]Saved:[/green] {output_path}", expand=False))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sequence-length experiment plots from W&B experiment data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python seq_len_plots.py -p 16_04_2026_seq_len_experiments_phme20_prognostics_raw\n"
            "  python seq_len_plots.py -p 16_04_2026_seq_len_experiments_concepts_n_cmapss_ds02_prognostics_ -u my-wandb-user -o ./out\n"
            "  python seq_len_plots.py --all\n"
        ),
    )
    parser.add_argument("--project", "-p", default=None, help="W&B project name")
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
    parser.add_argument(
        "--errorbar",
        default="sd",
        help=(
            "Seaborn errorbar style (default: sd). "
            "Examples: sd, se, ci, pi, 'pi,95', 'ci,95', none"
        ),
    )
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
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
            run_project(project, args.user, args.output_dir, args.no_summary, errorbar=_parse_errorbar(args.errorbar), no_title=args.no_title)
        except Exception as exc:
            console.print(f"[red]FAILED {project}: {exc}[/red]")
            failed.append((project, str(exc)))

    if failed:
        console.print(
            Panel(
                "\n".join(f"[red]{p}[/red]: {e}" for p, e in failed),
                title=f"[red]{len(failed)} project(s) failed[/red]",
                expand=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
