"""CLI for plotting training class balance vs. subset ratio for MZVAV diagnostics.

Runs the full MZVAV preprocessing pipeline (datasource → transforms → dataset)
for each subset ratio and reads the label distribution from the fit-predict table
that the model actually receives.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "configs"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

console = Console()

logging.getLogger("picid").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)

# MZVAV class labels (from MZVAV.py fault_number_by_dates)
CLASS_NAMES = {0: "Normal", 1: "Damper", 2: "Heating Coil", 3: "Cooling Coil"}
CLASS_COLORS = {
    0: "#2166ac",
    1: "#d73027",
    2: "#fc8d59",
    3: "#91bfdb",
}
NUM_CLASSES = 4

DEFAULT_SUBSET_RATIOS = [
    0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]
DEFAULT_SUBSET_BLOCKS = 3
DEFAULT_SUBSET_SEED = 0


def _compose_cfg(data_dir: str, subset_ratio: float, subset_blocks: int | None, subset_seed: int):
    """Compose the full Hydra config for MZVAV diagnostics with the given subset params."""
    import hydra
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra

    os.environ.setdefault("PROJECT_ROOT", str(_REPO_ROOT))

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=str(_CONFIG_DIR),
    ):
        cfg = compose(
            config_name="run",
            overrides=[
                "experiment=mzvav/diagnostics/tabpfn_fit_predict",
                f"paths.data_dir={data_dir}",
                f"+task_definition.subset_ratio={subset_ratio}",
                f"+task_definition.subset_seed={subset_seed}",
                *(
                    [f"+task_definition.subset_blocks={subset_blocks}"]
                    if subset_blocks is not None
                    else []
                ),
                "hydra/job_logging=default",
                "hydra/hydra_logging=default",
            ],
        )
    return cfg


def get_labels_for_ratio(
    data_dir: str,
    subset_ratio: float,
    subset_blocks: int | None,
    subset_seed: int,
) -> np.ndarray:
    """Run the full MZVAV pipeline and return the flat training label array."""
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from picid.transforms.base.transform_manager import ConfigTransformManager
    from picid.data.preprocessing.preprocessor import PreProcessor
    from picid.data.data_objects import SplitViewPolicy

    cfg = _compose_cfg(data_dir, subset_ratio, subset_blocks, subset_seed)

    # 1. Datasource
    datasource = hydra.utils.instantiate(cfg.datasource)

    # 2. Transforms (reads subset_ratio from task_definition via OmegaConf interpolation)
    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)

    # 3. Full preprocessing pipeline (direct mode, no caching)
    preprocessor = PreProcessor(datasource=datasource, transforms=transforms_manager)
    preprocessor.pipeline()

    data_dict = preprocessor.get_processed_split_dict(SplitViewPolicy.KEEP_UNIT_LISTS)
    meta_data_dict = preprocessor.get_meta_data_dict()

    # 4. Dataset — the actual fit-predict table (batch_size=1 per task)
    input_keys = list(cfg.task_definition.model.data_requirements.input_tensors)
    dataset_train = hydra.utils.instantiate(
        cfg.dataset,
        data_dict={k: data_dict["train"][k] for k in input_keys},
        meta_data_dict=meta_data_dict,
    )

    # 5. Get the full training table from the single task
    # dataset[[0]] returns AttributeDict with:
    #   target:  (1, n_sequences, 1)  — class labels the model sees
    #   context: (1, n_sequences, n_features)
    batch = dataset_train[[0]]
    y = batch.target.numpy()          # (1, n_sequences, 1)
    labels = y.reshape(-1).astype(int)

    GlobalHydra.instance().clear()
    return labels


def compute_class_counts(labels: np.ndarray) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=int)
    for cls in range(NUM_CLASSES):
        counts[cls] = int(np.sum(labels == cls))
    return counts


def display_counts_table(
    subset_ratios: list[float],
    counts_per_ratio: dict[float, np.ndarray],
) -> None:
    table = Table(title="Class counts per subset ratio", show_lines=True)
    table.add_column("Subset Ratio", style="bold", no_wrap=True)
    for cls in range(NUM_CLASSES):
        table.add_column(CLASS_NAMES[cls], justify="right")
    table.add_column("Total", justify="right", style="bold")

    for ratio in subset_ratios:
        counts = counts_per_ratio[ratio]
        total = int(counts.sum())
        cells = []
        for cls in range(NUM_CLASSES):
            c = int(counts[cls])
            cells.append("[yellow]0[/yellow]" if c == 0 else str(c))
        table.add_row(f"{ratio:.4g}", *cells, str(total))

    console.print(table)


def generate_plot(
    subset_ratios: list[float],
    counts_per_ratio: dict[float, np.ndarray],
    output_dir: str,
    subset_blocks: int | None = None,
    subset_seed: int = 0,
    no_title: bool = False,
) -> Path:
    x_pos = np.arange(len(subset_ratios))
    x_labels = [f"{r:.4g}" for r in subset_ratios]

    with sns.plotting_context("paper", font_scale=2, rc={"font.family": "Helvetica"}):
        fig, ax = plt.subplots(figsize=(14, 7))

        bottoms = np.zeros(len(subset_ratios))
        for cls in range(NUM_CLASSES):
            class_counts = np.array([counts_per_ratio[r][cls] for r in subset_ratios])
            ax.bar(
                x_pos,
                class_counts,
                bottom=bottoms,
                label=CLASS_NAMES[cls],
                color=CLASS_COLORS[cls],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += class_counts

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel("Subset Ratio")
        ax.set_ylabel("Number of Training Samples")
        if not no_title:
            ax.set_title("Training Class Balance vs. Subset Ratio [Dataset: MZVAV]")
        ax.set_axisbelow(True)
        ax.grid(axis="y")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

        ax.tick_params(axis="x", width=1.5)
        ax.tick_params(axis="y", width=1.5)

        legend = ax.legend(title="", loc="upper left", framealpha=1)

        fig.tight_layout()

        blocks_tag = f"blocks{subset_blocks}" if subset_blocks is not None else "blocks_none"
        output_path = Path(output_dir) / f"class_balance_mzvav_subset_ratio_{blocks_tag}_seed{subset_seed}.pdf"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training class balance vs. subset ratio for MZVAV diagnostics. "
            "Runs the full preprocessing pipeline (datasource → transforms → dataset) "
            "and reads labels from the fit-predict table the model sees."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python class_balance_plot.py --data-dir ~/datasets\n"
            "  python class_balance_plot.py --data-dir ~/datasets --subset-blocks 3 -o ./out\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        required=True,
        help="Path to the datasets root directory (parent of 'building/').",
    )
    parser.add_argument(
        "--subset-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_SUBSET_RATIOS,
        metavar="R",
        help=f"Subset ratios to evaluate (default: {DEFAULT_SUBSET_RATIOS}).",
    )
    parser.add_argument(
        "--subset-blocks",
        type=lambda v: None if v.lower() == "none" else int(v),
        default=DEFAULT_SUBSET_BLOCKS,
        metavar="N",
        help=(
            f"Number of contiguous blocks for block-based subsetting "
            f"(default: {DEFAULT_SUBSET_BLOCKS}). Pass 'none' to use plain subset_ratio "
            f"without block structure."
        ),
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=DEFAULT_SUBSET_SEED,
        help=f"Random seed for subset sampling (default: {DEFAULT_SUBSET_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="picid_report_artifacts",
        help="Output directory for the PDF (default: picid_report_artifacts).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing the counts table.",
    )
    parser.add_argument(
        "--no-title", action="store_true", help="Omit plot title (for publication figures)"
    )
    args = parser.parse_args()

    data_dir = str(Path(args.data_dir).expanduser().resolve())

    console.print(
        Panel(
            f"[bold]Data dir:[/bold]      {data_dir}\n"
            f"[bold]Subset ratios:[/bold] {args.subset_ratios}\n"
            f"[bold]Subset blocks:[/bold] {args.subset_blocks}\n"
            f"[bold]Subset seed:[/bold]   {args.subset_seed}\n"
            f"[bold]Output:[/bold]        {args.output_dir}",
            title="Class Balance Plot – MZVAV",
            expand=False,
        )
    )

    counts_per_ratio: dict[float, np.ndarray] = {}

    for ratio in args.subset_ratios:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            t = progress.add_task(
                f"Running full pipeline for subset_ratio={ratio:.4g}...", total=None
            )
            labels = get_labels_for_ratio(
                data_dir=data_dir,
                subset_ratio=ratio,
                subset_blocks=args.subset_blocks,
                subset_seed=args.subset_seed,
            )
            counts = compute_class_counts(labels)
            counts_per_ratio[ratio] = counts
            progress.update(
                t,
                description=(
                    f"[green]ratio={ratio:.4g}: "
                    + ", ".join(
                        f"{CLASS_NAMES[c]}={counts[c]}" for c in range(NUM_CLASSES)
                    )
                    + f" (total={counts.sum()})"
                ),
            )

    if not args.no_summary:
        display_counts_table(args.subset_ratios, counts_per_ratio)

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        t = progress.add_task("Generating plot...", total=None)
        output_path = generate_plot(args.subset_ratios, counts_per_ratio, args.output_dir, subset_blocks=args.subset_blocks, subset_seed=args.subset_seed, no_title=args.no_title)
        progress.update(t, description="[green]Plot generated.")

    console.print(Panel(f"[green]Saved:[/green] {output_path}", expand=False))


if __name__ == "__main__":
    main()
