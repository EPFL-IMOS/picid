"""CLI for plotting TabPFN quantile/RUL distribution heatmaps from prediction files."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = Path("~/Experiments/TabPHM").expanduser()
FALLBACK_INPUTS = [
    _REPO_ROOT / "lmetk_report_artifacts" / "predictions_LB20.pt",
    _REPO_ROOT / "lmetk_report_artifacts" / "predictions_LB1.pt",
]
DEFAULT_OUTPUT_DIR = "picid_report_artifacts"
DEFAULT_OUTPUT_NAME = "quantile_distribution_heatmap"
DEFAULT_BIN_START = 1500
DEFAULT_BIN_STOP = 3500
DEFAULT_NUM_BINS = 100
DEFAULT_MAX_YTICKS = 10


@dataclass
class DistributionHeatmapData:
    path: Path
    label: str
    probabilities: np.ndarray
    bucket_means: np.ndarray
    original_time_steps: int
    selected_bin_count: int


@dataclass(frozen=True)
class PredictionFile:
    path: Path
    project: str | None
    lookback: int | None


def _parse_prediction_file(path: Path) -> PredictionFile:
    stem = path.stem
    prefixed = re.fullmatch(
        r"(?P<project>.+?)_predictions_(?P<lookback>\d+)",
        stem,
        flags=re.IGNORECASE,
    )
    if prefixed:
        return PredictionFile(
            path=path,
            project=prefixed.group("project"),
            lookback=int(prefixed.group("lookback")),
        )

    legacy = re.search(r"LB(?P<lookback>\d+)", stem, flags=re.IGNORECASE)
    return PredictionFile(
        path=path,
        project=None,
        lookback=int(legacy.group("lookback")) if legacy else None,
    )


def _discover_default_inputs(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.expanduser().glob("*_predictions_*.pt"))
    if paths:
        return paths
    return FALLBACK_INPUTS


def _project_sort_key(project: str | None) -> tuple[int, str]:
    if project is None:
        return (1, "")
    return (0, project.lower())


def _file_sort_key(item: PredictionFile) -> tuple[int, int, str]:
    # Higher lookback first, matching the original notebook panel order.
    lookback_key = -item.lookback if item.lookback is not None else 0
    return (_project_sort_key(item.project)[0], lookback_key, item.path.name)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _project_display_name(project: str | None) -> str:
    if project is None:
        return "default"
    names = {
        "phme20": "PHME20",
        "unibo": "Unibo",
    }
    return names.get(project.lower(), project.replace("_", " ").title())


def _infer_label(path: Path) -> str:
    parsed = _parse_prediction_file(path)
    if parsed.lookback is not None:
        lookback = parsed.lookback
        unit = "step" if lookback == 1 else "steps"
        return f"{lookback} lookback {unit}"
    return path.stem.replace("_", " ")


def _validate_labels(
    files: list[PredictionFile], labels: list[str] | None
) -> list[str]:
    if labels is None:
        return [_infer_label(item.path) for item in files]
    if len(labels) != len(files):
        raise ValueError(
            f"Expected {len(files)} labels for {len(files)} inputs, got {len(labels)}."
        )
    return labels


def _group_files(
    paths: list[Path],
    labels: list[str] | None,
) -> dict[str | None, list[tuple[Path, str]]]:
    files = [_parse_prediction_file(path.expanduser().resolve()) for path in paths]
    resolved_labels = _validate_labels(files, labels)
    grouped_files = sorted(
        zip(files, resolved_labels),
        key=lambda pair: (_project_sort_key(pair[0].project), _file_sort_key(pair[0])),
    )

    groups: dict[str | None, list[tuple[Path, str]]] = {}
    for item, label in grouped_files:
        groups.setdefault(item.project, []).append((item.path, label))
    return groups


def _downsample_bins(
    probabilities: np.ndarray,
    bucket_means: np.ndarray,
    num_bins: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_bins is None or num_bins <= 0 or probabilities.shape[1] <= num_bins:
        return probabilities, bucket_means

    groups = np.array_split(np.arange(probabilities.shape[1]), num_bins)
    probabilities_down = np.stack(
        [probabilities[:, group].mean(axis=1) for group in groups],
        axis=1,
    )
    bucket_means_down = np.array(
        [bucket_means[group].mean() for group in groups],
        dtype=bucket_means.dtype,
    )
    return probabilities_down, bucket_means_down


def load_distribution_heatmap_data(
    path: Path,
    label: str,
    bin_start: int,
    bin_stop: int | None,
    num_bins: int | None,
    time_stride: int,
) -> DistributionHeatmapData:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dict, got {type(payload)!r}.")
    if "logits" not in payload or "criterion" not in payload:
        raise KeyError(f"{path} must contain 'logits' and 'criterion' entries.")

    logits = payload["logits"].detach().cpu()
    criterion = payload["criterion"]
    borders = criterion.borders.detach().cpu()
    bucket_widths = criterion.bucket_widths.detach().cpu()

    if logits.ndim != 2:
        raise ValueError(f"{path} logits must be 2D, got shape {tuple(logits.shape)}.")
    if len(bucket_widths) != logits.shape[1]:
        raise ValueError(
            f"{path} has {logits.shape[1]} logit bins but {len(bucket_widths)} bucket widths."
        )
    if time_stride < 1:
        raise ValueError("time_stride must be >= 1.")

    n_bins = logits.shape[1]
    bin_stop = n_bins if bin_stop is None else bin_stop
    if not (0 <= bin_start < bin_stop <= n_bins):
        raise ValueError(
            f"Invalid bin range [{bin_start}, {bin_stop}) for {n_bins} bins in {path}."
        )

    bucket_means = borders[:-1] + bucket_widths / 2
    selected_indices = torch.arange(n_bins - 1 - bin_start, n_bins - bin_stop - 1, -1)

    with torch.no_grad():
        logits = logits[::time_stride]
        normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
        selected_logits = logits.index_select(1, selected_indices)
        probabilities = torch.exp(selected_logits - normalizer).numpy()

    selected_bucket_means = bucket_means.index_select(0, selected_indices).numpy()
    probabilities, selected_bucket_means = _downsample_bins(
        probabilities,
        selected_bucket_means,
        num_bins,
    )

    return DistributionHeatmapData(
        path=path,
        label=label,
        probabilities=probabilities,
        bucket_means=selected_bucket_means,
        original_time_steps=int(payload["logits"].shape[0]),
        selected_bin_count=int(bin_stop - bin_start),
    )


def plot_distribution_heatmaps(
    heatmaps: list[DistributionHeatmapData],
    output_dir: Path,
    output_name: str,
    formats: list[str],
    title: str,
    no_title: bool,
    cmap: str,
    max_yticks: int,
    y_label: str,
    x_label: str,
) -> list[Path]:
    if not heatmaps:
        raise ValueError("At least one heatmap is required.")

    nrows = len(heatmaps)
    figsize = (28, max(4.5, 4.8 * nrows))

    with sns.plotting_context("paper", font_scale=2, rc={"font.family": "Helvetica"}):
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, squeeze=False)

        for ax, data in zip(axes[:, 0], heatmaps):
            sns.heatmap(
                data.probabilities.T,
                yticklabels=False,
                xticklabels=False,
                cmap=cmap,
                ax=ax,
                cbar=True,
                cbar_kws={"label": "Probability"},
            )
            ax.set_title(data.label)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            tick_count = min(max_yticks, len(data.bucket_means))
            if tick_count > 0:
                tick_indices = np.linspace(
                    0,
                    len(data.bucket_means) - 1,
                    tick_count,
                    dtype=int,
                )
                ax.set_yticks(tick_indices + 0.5)
                ax.set_yticklabels(
                    [f"{data.bucket_means[i]:.2f}" for i in tick_indices]
                )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_color("black")

        if not no_title:
            fig.suptitle(title)

        fig.tight_layout()
        if not no_title:
            fig.subplots_adjust(top=0.93)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for fmt in formats:
            output_path = output_dir / f"{output_name}.{fmt}"
            fig.savefig(
                output_path,
                dpi=300,
                transparent=True,
                bbox_inches="tight",
                format=fmt,
            )
            output_paths.append(output_path)

        plt.close(fig)

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate quantile/RUL distribution heatmaps from TabPFN prediction .pt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python quantile_distribution_plots.py\n"
            "  python quantile_distribution_plots.py --input-dir ~/Experiments/TabPHM\n"
            "  python quantile_distribution_plots.py lmetk_report_artifacts/predictions_LB20.pt lmetk_report_artifacts/predictions_LB1.pt\n"
            '  python quantile_distribution_plots.py --labels "20 lookback steps" "1 lookback step" --formats pdf png\n'
        ),
    )
    parser.add_argument(
        "prediction_files",
        nargs="*",
        type=Path,
        help=(
            "Prediction .pt files. If omitted, files are discovered from --input-dir "
            "and grouped by the prefix before '_predictions_'."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Default directory scanned for '*_predictions_*.pt' files "
            f"(default: {DEFAULT_INPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Panel labels, one per prediction file. Defaults are inferred from filenames.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output filename without extension (default: {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["pdf", "png", "svg"],
        help="Output formats to save (default: png).",
    )
    parser.add_argument(
        "--bin-start",
        type=int,
        default=DEFAULT_BIN_START,
        help=f"Start index after reversing buckets (default: {DEFAULT_BIN_START}).",
    )
    parser.add_argument(
        "--bin-stop",
        type=int,
        default=DEFAULT_BIN_STOP,
        help=f"Stop index after reversing buckets (default: {DEFAULT_BIN_STOP}).",
    )
    parser.add_argument(
        "--num-bins",
        type=lambda value: None if value.lower() == "none" else int(value),
        default=DEFAULT_NUM_BINS,
        help=f"Number of displayed RUL bins after downsampling (default: {DEFAULT_NUM_BINS}).",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=1,
        help="Plot every Nth time step to reduce figure size (default: 1).",
    )
    parser.add_argument(
        "--max-yticks",
        type=int,
        default=DEFAULT_MAX_YTICKS,
        help=f"Maximum number of RUL tick labels (default: {DEFAULT_MAX_YTICKS}).",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name (default: viridis).",
    )
    parser.add_argument(
        "--title",
        default="RUL Distribution over Time Steps",
        help="Figure title.",
    )
    parser.add_argument(
        "--x-label",
        default="Time Steps (all cycles in test set)",
        help="X-axis label.",
    )
    parser.add_argument(
        "--y-label",
        default="RUL (not rescaled)",
        help="Y-axis label.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the figure title (for publication figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.prediction_files or _discover_default_inputs(args.input_dir)
    groups = _group_files(paths, args.labels)

    console.print(
        Panel(
            "\n".join(
                [
                    f"[bold]Inputs:[/bold]       {sum(len(v) for v in groups.values())} files",
                    f"[bold]Projects:[/bold]     {', '.join(_project_display_name(project) for project in groups)}",
                    f"[bold]Input dir:[/bold]    {args.input_dir.expanduser()}",
                    f"[bold]Bin range:[/bold]    [{args.bin_start}, {args.bin_stop})",
                    f"[bold]Display bins:[/bold] {args.num_bins}",
                    f"[bold]Time stride:[/bold]  {args.time_stride}",
                    f"[bold]Output base:[/bold]  {args.output_dir / args.output_name}",
                ]
            ),
            title="Quantile Distribution Plots",
            expand=False,
        )
    )

    all_output_paths: list[Path] = []
    for project, project_files in groups.items():
        project_name = _project_display_name(project)
        output_name = args.output_name
        if project is not None:
            output_name = f"{args.output_name}_{_safe_slug(project)}"

        console.print(
            Panel(
                "\n".join(f"{path.name} -> {label}" for path, label in project_files),
                title=f"Project: {project_name}",
                expand=False,
            )
        )

        heatmaps: list[DistributionHeatmapData] = []
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            task = progress.add_task(
                f"Loading {project_name} prediction files...",
                total=len(project_files),
            )
            for path, label in project_files:
                progress.update(task, description=f"Loading {path.name}...")
                heatmaps.append(
                    load_distribution_heatmap_data(
                        path=path,
                        label=label,
                        bin_start=args.bin_start,
                        bin_stop=args.bin_stop,
                        num_bins=args.num_bins,
                        time_stride=args.time_stride,
                    )
                )
                progress.advance(task)

        for data in heatmaps:
            console.print(
                f"[green]{data.path.name}[/green]: "
                f"{data.original_time_steps} time steps, "
                f"{data.selected_bin_count} selected bins -> "
                f"{data.probabilities.shape[1]} displayed bins"
            )

        title = args.title
        if project is not None:
            title = f"{args.title} [{project_name}]"

        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            task = progress.add_task(f"Generating {project_name} plot...", total=None)
            output_paths = plot_distribution_heatmaps(
                heatmaps=heatmaps,
                output_dir=args.output_dir,
                output_name=output_name,
                formats=args.formats,
                title=title,
                no_title=args.no_title,
                cmap=args.cmap,
                max_yticks=args.max_yticks,
                y_label=args.y_label,
                x_label=args.x_label,
            )
            all_output_paths.extend(output_paths)
            progress.update(task, description="[green]Plot generated.")

    console.print(
        Panel(
            "\n".join(f"[green]Saved:[/green] {path}" for path in all_output_paths),
            expand=False,
        )
    )


if __name__ == "__main__":
    main()
