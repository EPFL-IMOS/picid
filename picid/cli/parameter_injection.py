"""Parameter injection: extract overridable params from experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from picid.cli.config_discovery import CONFIGS_ROOT


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _extract_optimization_from_defaults(defaults: list[Any]) -> str | None:
    """Extract optimization config name from a defaults list."""
    for item in defaults:
        if isinstance(item, dict) and "override /optimization" in item:
            opt_val = item["override /optimization"]
            if isinstance(opt_val, str) and opt_val:
                return opt_val
        elif isinstance(item, str) and "override /optimization:" in item:
            opt_part = item.split("override /optimization:")[-1].strip()
            if opt_part:
                return opt_part
    return None


def get_model_specific_params(model_name: str, task_type: str) -> dict[str, Any]:
    """Load configs/model_configs/<task_type>/<model_name>.yaml, return the loaded dict.

    Returns {} if the file does not exist.
    """
    path = CONFIGS_ROOT / "model_configs" / task_type / f"{model_name}.yaml"
    if not path.exists():
        return {}
    return load_yaml(path)


def get_overridable_from_experiment(experiment_key: str) -> dict[str, Any]:
    """Parse experiment YAML defaults to find base and model_configs.

    Returns dict with:
        - optimization: str, the effective optimization config name (e.g. default, sgd, reduce_on_plateau)
        - model_params: dict with overridable params (e.g. lr from optimization config)
    """
    exp_path = CONFIGS_ROOT / "experiment" / f"{experiment_key}.yaml"
    if not exp_path.exists():
        return {"optimization": "default", "model_params": {}}

    exp_cfg = load_yaml(exp_path)
    defaults = exp_cfg.get("defaults", [])

    optimization = "default"

    # Check experiment defaults directly
    found = _extract_optimization_from_defaults(defaults)
    if found:
        optimization = found
    else:
        # Traverse configs from defaults (base, model_configs, etc.)
        for item in defaults:
            if not isinstance(item, str) or item.startswith("override /"):
                continue
            if item.startswith("/"):
                cfg_path = CONFIGS_ROOT / f"{item.lstrip('/')}.yaml"
            else:
                cfg_path = CONFIGS_ROOT / "experiment" / f"{item}.yaml"
            if not cfg_path.exists():
                continue
            sub_cfg = load_yaml(cfg_path)
            sub_defaults = sub_cfg.get("defaults", [])
            found = _extract_optimization_from_defaults(sub_defaults)
            if found:
                optimization = found
                break

    model_params: dict[str, Any] = {}
    opt_path = CONFIGS_ROOT / "optimization" / f"{optimization}.yaml"
    if opt_path.exists():
        opt_cfg = load_yaml(opt_path)
        if "lr" in opt_cfg:
            model_params["lr"] = opt_cfg["lr"]

    return {"optimization": optimization, "model_params": model_params}
