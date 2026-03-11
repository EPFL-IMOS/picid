"""
Generate reproduction scripts from a report output and experiment .sh scripts.

Reads best HP config from report_output/.../tables/hp_impact/*.csv, maps report (dataset, model)
to script dataset string and model name, parses the experiment script(s) for the exact command
shape, and writes one scripts/reproduce/<dataset>_<model>_reproduce.sh per (dataset, model)
with WandB overridden to a reproduction project so runs do not pollute the HP-search project.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from picid_report.scripts.load_best_config import load_best_configs_from_report
from picid_report.scripts.parse_experiment_script import parse_experiment_script, ParsedScript


# Default seeds for baselines when not parsed from script (paper_prognostics/baselines.sh)
DEFAULT_BASELINE_SEEDS = [72, 88, 101, 666, 226688]

# Mapping from (report_dataset, report_model_from_filename) -> (script_dataset_string, script_model_name, script_key)
# script_key: "baselines" | "fit_predict" to select which experiment script template to use.
REPRODUCE_MAPPING: Dict[Tuple[str, str], Tuple[str, str, str]] = {
    # UNIBO21 + baselines (paper_prognostics)
    ("UNIBO21", "baselines.lstm_model.LSTM_Forecaster"): ("unibo|combined|prognostics", "lstm", "baselines"),
    ("UNIBO21", "baselines.crossformer_model.Crossformer_Forecaster"): ("unibo|combined|prognostics", "crossformer", "baselines"),
    ("UNIBO21", "baselines.patchtst_model.PatchTST_Forecaster"): ("unibo|combined|prognostics", "patchtst", "baselines"),
    ("UNIBO21", "baselines.spacetimeformer_model.Spacetimeformer_Forecaster"): ("unibo|combined|prognostics", "stf", "baselines"),
    ("UNIBO21", "baselines.tide_model.TiDE_Forecaster"): ("unibo|combined|prognostics", "tide", "baselines"),
    ("UNIBO21", "baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster"): ("unibo|combined|prognostics", "timeseries_transformer", "baselines"),
    ("UNIBO21", "model.wrappers.cnn1d_wrapper.CNN1D_Wrapper"): ("unibo|combined|prognostics", "cnn_1d", "baselines"),
    ("UNIBO21", "model.wrappers.mlp_wrapper.MLPWrapper"): ("unibo|combined|prognostics", "mlp", "baselines"),
    ("UNIBO21", "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper__exponentia"): ("unibo|combined|prognostics", "exponential_regression", "baselines"),
    ("UNIBO21", "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper__linear_"): ("unibo|combined|prognostics", "linear_regression", "baselines"),
    ("UNIBO21", "model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper"): ("unibo|combined|prognostics", "tabdpt_fit_predict", "fit_predict"),
    ("UNIBO21", "model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper"): ("unibo|combined|prognostics", "tabpfn_fit_predict", "fit_predict"),
}


def _safe_filename(s: str) -> str:
    """Filesystem-safe name: replace | and / with _."""
    return re.sub(r"[\|/]", "_", s).strip("_")


def _build_exp_name(dataset_key: str, task_type: str, subexp: str, model_name: str) -> str:
    if subexp:
        return f"{dataset_key}/{task_type}/{subexp}/{model_name}"
    return f"{dataset_key}/{task_type}/{model_name}"


def _substitute_command(
    command_lines: List[str],
    context: Dict[str, Any],
    best_hp: Dict[str, Any],
    wandb_reproduce: str,
) -> List[str]:
    """
    Substitute placeholders in command lines. context and best_hp provide values;
    final_log_folder is overridden to wandb_reproduce_... so we pass it via context.
    """
    out: List[str] = []
    for line in command_lines:
        if line.strip().startswith("#"):
            out.append(line)
            continue
        # Replace ${var} with value from context or best_hp
        def repl(m: re.Match) -> str:
            var = m.group(1)
            if var in context:
                return str(context[var])
            if var in best_hp:
                return str(best_hp[var])
            # Map script var names to best_hp keys
            if var == "context_length" and "task_definition.seq_len" in best_hp:
                return str(best_hp["task_definition.seq_len"])
            if var == "train_set_stride" and "task_definition.stride_train" in best_hp:
                return str(best_hp["task_definition.stride_train"])
            if var == "seq_len" and "task_definition.seq_len" in best_hp:
                return str(best_hp["task_definition.seq_len"])
            if var == "max_learning_rate" and "optimization.lr" in best_hp:
                return str(best_hp["optimization.lr"])
            if var == "batch_size" and "datamodule.train_batch_size" in best_hp:
                return str(best_hp["datamodule.train_batch_size"])
            if var == "seed":
                return str(best_hp.get("seed", context.get("seed", 72)))
            return m.group(0)

        new_line = re.sub(r"\$\{(\w+)\}", repl, line)
        out.append(new_line)
    return out


def _default_baseline_hp(best_hp: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure baselines have batch_size, seed etc. when not in best row."""
    out = dict(best_hp)
    if "datamodule.train_batch_size" not in out:
        out["datamodule.train_batch_size"] = 512
    if "seed" not in out:
        out["seed"] = 72
    return out


def generate_one_reproduce_script(
    dataset_report: str,
    model_report: str,
    script_dataset: str,
    script_model: str,
    script_key: str,
    parsed: ParsedScript,
    best_hp: Dict[str, Any],
    output_path: Path,
    wandb_reproduce_prefix: str,
    script_dir_relative: str,
) -> None:
    """Write one reproduction .sh file for the given (dataset, model)."""
    # script_dataset is "dataset_key|subexp|task_type" (e.g. unibo|combined|prognostics)
    parts = script_dataset.split("|")
    dataset_key = parts[0] if parts else ""
    subexp = parts[1] if len(parts) >= 2 else ""
    task_type = parts[2] if len(parts) >= 3 else ""

    exp_name = _build_exp_name(dataset_key, task_type, subexp, script_model)
    dataset_run_name = dataset_key.replace("/", "_")
    if script_key == "baselines":
        best_hp = _default_baseline_hp(best_hp)
        seeds: List[int] = parsed.seeds if parsed.seeds else DEFAULT_BASELINE_SEEDS
        use_seed_loop = True
    else:
        seed = best_hp.get("seed", 72)
        use_seed_loop = False

    original_wandb = parsed.wandb_log_folder or "reproduce"
    wandb_reproduce = f"{wandb_reproduce_prefix}{original_wandb}"
    log_parts = [wandb_reproduce, dataset_run_name, task_type]
    if subexp:
        log_parts.append(subexp)
    final_log_folder = "_".join(p for p in log_parts if p)

    if use_seed_loop:
        run_name_tpl = f"001_{dataset_run_name}_{script_model}_{task_type}_{subexp}_${{seed}}"
        context_seed = "${seed}"
    else:
        run_name_tpl = f"001_{dataset_run_name}_{script_model}_{task_type}_{subexp}_{seed}"
        context_seed = seed

    context: Dict[str, Any] = {
        "exp_name": exp_name,
        "final_log_folder": final_log_folder,
        "run_name": run_name_tpl,
        "dataset_key": dataset_key,
        "subexp": subexp,
        "task_type": task_type,
        "model_name": script_model,
        "dataset_run_name": dataset_run_name,
        "seed": context_seed,
        "check_val_every_n_epoch": 1,
        "device": 0,
        "use_preprocessing_file_lock": "False",
        "max_epochs": 200,
    }

    # For seed loop, best_hp must not override seed (we use ${seed} in command)
    hp_for_cmd = dict(best_hp)
    if use_seed_loop:
        hp_for_cmd.pop("seed", None)

    command_lines = _substitute_command(parsed.command_lines, context, hp_for_cmd, wandb_reproduce)
    # Drop comment-only lines so the bash array is a clean list of args
    command_lines = [c for c in command_lines if not c.strip().startswith("#")]

    lines = [
        "#!/bin/bash",
        "",
        "# Reproduce best run from report (same best HPs, all seeds).",
        f"# Dataset: {dataset_report} -> {script_dataset}, Model: {model_report} -> {script_model}",
        "",
        "LOG_DIR=\"./logs\"",
        "mkdir -p \"$LOG_DIR\"",
        "SCRIPT_DIR=\"$( cd \"$( dirname \"${BASH_SOURCE[0]}\" )\" &> /dev/null && pwd )\"",
        "# Source logging from project scripts (relative to scripts/reproduce -> scripts/base)",
        f"source \"$SCRIPT_DIR/{script_dir_relative}\"",
        "",
        "export HYDRA_FULL_ERROR=1",
        f"wandb_log_folder=\"{wandb_reproduce}\"",
        "DEBUG_SKIP=\"${DEBUG_SKIP:-}\"",
        "",
        f"dataset=\"{script_dataset}\"",
        f"model_name=\"{script_model}\"",
        f"exp_name=\"{exp_name}\"",
        f"final_log_folder=\"{final_log_folder}\"",
        f"dataset_run_name=\"{dataset_run_name}\"",
        "",
    ]

    if use_seed_loop:
        seeds_str = " ".join(str(s) for s in seeds)
        lines.append(f"SEEDS=({seeds_str})")
        lines.append("")
        lines.append("for seed in \"${SEEDS[@]}\"; do")
        lines.append(f'  run_name="001_{dataset_run_name}_{script_model}_{task_type}_{subexp}_${{seed}}"')
        lines.append("")
        lines.append("  command_to_run=(")
        for cl in command_lines:
            lines.append("    " + cl + " \\")
        if lines[-1].endswith(" \\"):
            lines[-1] = lines[-1][:-2]
        lines.append("  )")
        lines.append("")
        lines.append('  run_and_log "${command_to_run[*]}" "$run_name" "$DEBUG_SKIP"')
        lines.append("done")
    else:
        lines.append(f"run_name=\"{run_name_tpl}\"")
        lines.append("")
        lines.append("command_to_run=(")
        for cl in command_lines:
            lines.append("    " + cl + " \\")
        if lines[-1].endswith(" \\"):
            lines[-1] = lines[-1][:-2]
        lines.append(")")
        lines.append("")
        lines.append('run_and_log "${command_to_run[*]}" "$run_name" "$DEBUG_SKIP"')

    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    output_path.chmod(0o755)


def generate_reproduce_scripts(
    report_dir: str | Path,
    script_paths: List[str | Path],
    output_dir: str | Path,
    wandb_reproduce_prefix: str = "reproduce_",
    mapping: Optional[Dict[Tuple[str, str], Tuple[str, str, str]]] = None,
    script_dir_relative: str = "../base/logging.sh",
) -> List[Path]:
    """
    Generate one reproduction script per (dataset, model) that has a best config and mapping.

    Parameters
    ----------
    report_dir : path to report output (e.g. report_output/29_01_2026_unibo_prognostics_combined)
    script_paths : paths to experiment .sh scripts (e.g. [baselines.sh, fit_predict.sh])
    output_dir : directory to write scripts (e.g. scripts/reproduce)
    wandb_reproduce_prefix : prefix for WandB project so runs go to e.g. reproduce_29_01_2026
    mapping : optional override for (report_dataset, report_model) -> (script_dataset, script_model, script_key)
    script_dir_relative : path from scripts/reproduce/ to logging.sh (default ../base/logging.sh)

    Returns
    -------
    List of written script paths.
    """
    report_dir = Path(report_dir)
    output_dir = Path(output_dir)
    mapping = mapping or REPRODUCE_MAPPING

    best_configs = load_best_configs_from_report(report_dir)
    if not best_configs:
        return []

    parsed_by_key: Dict[str, ParsedScript] = {}
    for sp in script_paths:
        p = parse_experiment_script(sp)
        if p is None:
            continue
        # Infer script_key from path name (baselines vs fit_predict)
        name = Path(sp).stem.lower()
        if "baseline" in name:
            parsed_by_key["baselines"] = p
        elif "fit_predict" in name or "fitpredict" in name:
            parsed_by_key["fit_predict"] = p

    written: List[Path] = []
    for (dataset_report, model_report), best_hp in best_configs.items():
        key = (dataset_report, model_report)
        if key not in mapping:
            continue
        script_dataset, script_model, script_key = mapping[key]
        if script_key not in parsed_by_key:
            continue
        parsed = parsed_by_key[script_key]

        dataset_safe = _safe_filename(script_dataset)
        model_safe = _safe_filename(script_model)
        out_name = f"{dataset_safe}_{model_safe}_reproduce.sh"
        out_path = output_dir / out_name

        generate_one_reproduce_script(
            dataset_report=dataset_report,
            model_report=model_report,
            script_dataset=script_dataset,
            script_model=script_model,
            script_key=script_key,
            parsed=parsed,
            best_hp=best_hp,
            output_path=out_path,
            wandb_reproduce_prefix=wandb_reproduce_prefix,
            script_dir_relative=script_dir_relative,
        )
        written.append(out_path)

    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate reproduction scripts from report and experiment scripts.")
    parser.add_argument("--report-dir", type=str, default="report_output/29_01_2026_unibo_prognostics_combined", help="Report output directory")
    parser.add_argument("--script", type=str, action="append", dest="scripts", help="Path to experiment .sh (repeat for baselines and fit_predict)")
    parser.add_argument("--output-dir", type=str, default="scripts/reproduce", help="Output directory for reproduce scripts")
    parser.add_argument("--wandb-prefix", type=str, default="reproduce_", help="WandB project prefix for reproduction runs")
    args = parser.parse_args()

    script_paths = args.scripts or [
        "scripts/paper_prognostics/baselines.sh",
        "scripts/paper_prognostics/fit_predict.sh",
    ]

    written = generate_reproduce_scripts(
        report_dir=args.report_dir,
        script_paths=script_paths,
        output_dir=args.output_dir,
        wandb_reproduce_prefix=args.wandb_prefix,
        script_dir_relative="../base/logging.sh",
    )
    print(f"Wrote {len(written)} reproduction script(s) to {args.output_dir}/")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
