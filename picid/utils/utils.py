"""Utility functions for project setup, reproducibility, and task wrapping."""

import json
import shutil
import subprocess
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from picid.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _get_hydra_overrides() -> list[str]:
    """
    Return Hydra command-line overrides if Hydra is initialized.

    Returns
    -------
    list[str]
        Hydra task overrides, or an empty list when Hydra is unavailable.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return list(HydraConfig.get().overrides.task)
    except Exception:
        pass
    return []


def _write_reproduce_guide(cfg: DictConfig) -> None:
    """
    Write a reproduction guide into the run output directory.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for the current run.
    """
    if "paths" not in cfg or "output_dir" not in cfg.paths:
        return
    out_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root_raw = cfg.paths.get("root_dir", ".")
    root = Path(str(root_raw)).expanduser().resolve()
    if not root or str(root) == ".":
        root = Path.cwd()

    # Portable paths for reproducibility across machines
    try:
        out_dir_rel = out_dir.relative_to(root)
        output_dir_portable = f"${{PROJECT_ROOT}}/{out_dir_rel}"
    except ValueError:
        output_dir_portable = str(out_dir)

    overrides = _get_hydra_overrides()
    run_args = " ".join(overrides) if overrides else "<add your overrides>"
    experiment = cfg.get("experiment_group", "debug")
    exp_choice = getattr(
        getattr(getattr(cfg, "hydra", None), "runtime", None),
        "choices",
        None,
    )
    experiment_key = ""
    if exp_choice and hasattr(exp_choice, "experiment"):
        experiment_key = str(exp_choice.experiment)
    elif overrides:
        for o in overrides:
            if o.startswith("experiment="):
                experiment_key = o.split("=", 1)[1]
                break

    # Build debug config for VS Code / Cursor (portable via ${workspaceFolder})
    debug_config = {
        "name": f"Reproduce: {experiment_key or experiment}",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/picid/run.py",
        "console": "integratedTerminal",
        "args": overrides if overrides else ["experiment=null"],
        "cwd": "${workspaceFolder}",
        "env": {"PROJECT_ROOT": "${workspaceFolder}", "HYDRA_FULL_ERROR": "1"},
    }
    if "debug=default" not in run_args and "debug=" not in run_args:
        debug_config["args"] = list(debug_config["args"]) + ["debug=default"]

    run_dir_arg = output_dir_portable.replace("${PROJECT_ROOT}", "${workspaceFolder}")
    debug_config_from_run = {
        "name": f"Reproduce from config: {experiment_key or experiment}",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/reproducibility/reproduce_from_run.py",
        "console": "integratedTerminal",
        "args": [run_dir_arg],
        "cwd": "${workspaceFolder}",
        "env": {"PROJECT_ROOT": "${workspaceFolder}", "HYDRA_FULL_ERROR": "1"},
    }

    git_info = _get_git_info(root)
    lines = [
        "# How to Reproduce This Experiment",
        "",
        f"**Output directory:** `{output_dir_portable}`",
    ]
    if git_info.get("git_commit"):
        lines.append(
            f"**Git commit:** `{git_info['git_commit']}` (see run_metadata.yaml)"
        )
        lines.append("")
    lines.extend(
        [
            "## 1. Environment setup",
            "",
            "```bash",
            "export PROJECT_ROOT=/path/to/PICID  # set to your project root",
            "cd $PROJECT_ROOT",
            "# Install dependencies (use uv.lock from this run dir for exact versions):",
            f"# cp {output_dir_portable}/uv.lock . && uv sync",
            "uv sync",
            "# Download datasets if needed (e.g. PHME20):",
            "# ./scripts/datasets_setup.sh",
            "```",
            "",
            "## 2. Run the model",
            "",
            "**Option A – from repo configs** (uses current configs + overrides):",
            "",
            "```bash",
            "export PROJECT_ROOT=/path/to/PICID  # set to your project root",
            "cd $PROJECT_ROOT",
            f"uv run python picid/run.py {run_args}",
            "```",
            "",
            "**Option B – from this run's config** (exact config from logs, for version comparison):",
            "",
            "```bash",
            "export PROJECT_ROOT=/path/to/PICID  # set to your project root",
            "cd $PROJECT_ROOT",
            f"uv run python scripts/reproducibility/reproduce_from_run.py {output_dir_portable}",
            "```",
            "",
            "## 3. Debug configuration (VS Code / Cursor)",
            "",
            "Add to `.vscode/launch.json` under `configurations`:",
            "",
            "**Option A (from repo):**",
            "",
            "```json",
            json.dumps(debug_config, indent=2),
            "```",
            "",
            "**Option B (from run config):**",
            "",
            "```json",
            json.dumps(debug_config_from_run, indent=2),
            "```",
            "",
            "## Files in this run",
            "",
            "- `run_metadata.yaml` – git commit, branch, dirty flag",
            "- `config_resolved.yaml` – full resolved config",
            "- `uv.lock` – exact dependency versions",
            "- `checkpoints/` – model checkpoints",
            "- `logs/` – training logs",
            "",
        ]
    )
    dest = out_dir / "REPRODUCE.md"
    try:
        dest.write_text("\n".join(lines), encoding="utf-8")
        log.debug(f"Saved reproducibility guide: {dest}")
    except OSError as e:
        log.warning(f"Could not write REPRODUCE.md: {e}")


def _get_git_info(root: Path) -> dict[str, str | bool]:
    """
    Return git commit, branch, and dirty flag for reproducibility.

    Parameters
    ----------
    root : Path
        Repository root used as the git working directory.

    Returns
    -------
    dict[str, str | bool]
        Git metadata suitable for persistence.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=5,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=5,
        )
        return {
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else "",
            "git_branch": branch.stdout.strip() if branch.returncode == 0 else "",
            "git_dirty": (
                bool(status.stdout.strip()) if status.returncode == 0 else False
            ),
        }
    except Exception:
        return {"git_commit": "", "git_branch": "", "git_dirty": False}


def _save_git_info_to_run(cfg: DictConfig) -> None:
    """
    Save git commit, branch, and dirty state to run metadata.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for the current run.
    """
    if "paths" not in cfg or "output_dir" not in cfg.paths:
        return
    root_raw = cfg.paths.get("root_dir", ".")
    root = Path(str(root_raw)).expanduser().resolve()
    if not root or str(root) == ".":
        root = Path.cwd()
    out_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    info = _get_git_info(root)
    dest = out_dir / "run_metadata.yaml"
    try:
        OmegaConf.save(OmegaConf.create(info), dest)
        log.debug(f"Saved git info for reproducibility: {dest}")
    except OSError as e:
        log.warning(f"Could not write run_metadata.yaml: {e}")


def _save_uv_lock_to_run(cfg: DictConfig) -> None:
    """
    Copy ``uv.lock`` into the run output directory for reproducibility.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for the current run.
    """
    if "paths" not in cfg or "output_dir" not in cfg.paths:
        return
    root = Path(cfg.paths.get("root_dir", ".")).expanduser().resolve()
    uv_lock = root / "uv.lock"
    out_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    if not uv_lock.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "uv.lock"
    try:
        shutil.copy2(uv_lock, dest)
        log.debug(f"Saved uv.lock to run dir for reproducibility: {dest}")
    except OSError as e:
        log.warning(f"Could not copy uv.lock to run dir: {e}")


def extras(cfg: DictConfig) -> None:
    """
    Apply optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings.
        - Setting tags from command line.
        - Rich config printing.

    Parameters
    ----------
    cfg : DictConfig
        A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """
    Decorate a task function with reproducibility and cleanup helpers.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure).
    - save the exception to a `.log` file.
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later).
    - etc. (adjust depending on your needs).

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    Parameters
    ----------
    task_func : Callable
        The task function to be wrapped.

    Returns
    -------
    Callable
        The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Execute the wrapped task function with run bookkeeping.

        Parameters
        ----------
        cfg : DictConfig
            A DictConfig object containing the config tree.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            The metric and object dictionaries returned by the task function.
        """
        failed = False

        try:
            _save_uv_lock_to_run(cfg)
            _save_git_info_to_run(cfg)
            _write_reproduce_guide(cfg)
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            failed = True
            log.exception(f"Exception occurred while executing task: {ex}")

            if (
                "debug" in cfg
                and cfg.debug.raise_inner_exception in cfg.debug
                and cfg.debug.raise_inner_exception
            ):
                raise
            else:
                # some hyperparameter combinations might be invalid or cause out-of-memory errors
                # so when using hparam search plugins like Optuna, you might want to disable
                # raising the below exception to avoid multirun failure
                raise

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish(exit_code=1 if failed else 0)

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: dict[str, Any], metric_name: str | None
) -> float | None:
    """
    Safely retrieve the value of a named metric.

    Parameters
    ----------
    metric_dict : dict
        A dict containing metric values.
    metric_name : str, optional
        If provided, the name of the metric to retrieve.

    Returns
    -------
    float, None
        If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
