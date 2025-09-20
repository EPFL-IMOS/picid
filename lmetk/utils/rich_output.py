from pathlib import Path

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from lmetk.data.data_objects.base import BaseDataObject


def extract_targets(cfg, prefix=""):
    targets = []
    if not OmegaConf.is_dict(cfg):
        return targets

    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if OmegaConf.is_dict(v):
            if "_target_" in v:
                target_str = v["_target_"]
                if "." in target_str and not target_str.endswith(".py"):
                    # Try to resolve to a Python file (best effort)
                    module_path = (
                        Path(target_str.replace(".", "/") + ".py").resolve().as_uri()
                    )
                else:
                    module_path = ""
                targets.append((full_key, target_str, module_path))
            # recurse into nested dict
            targets.extend(extract_targets(v, prefix=full_key))
    return targets


def display_targets(cfg):
    table = Table(title="Hydra _target_ Classes")
    table.add_column("Config Path", style="cyan", no_wrap=True)
    table.add_column("_target_", style="magenta")
    # table.add_column(
    #     "Resolved Module Path", style="green", overflow="fold", no_wrap=False
    # )

    for path, target, resolved in extract_targets(cfg):
        table.add_row(path, target)

    Console().print(table)


# Plot the data_dict structure with tensor dimensions
def print_data_dict_structure(data_dict):

    def describe(obj):
        if isinstance(obj, dict) or isinstance(obj, BaseDataObject):
            return "{...}"
        elif isinstance(obj, np.ndarray):
            return f"np.ndarray{obj.shape}"
        elif torch.is_tensor(obj):
            return f"torch.Tensor{tuple(obj.shape)}"
        elif isinstance(obj, list):
            if obj and (torch.is_tensor(obj[0]) or isinstance(obj[0], np.ndarray)):
                return f"list\\[{describe(obj[0])}\\] x {len(obj)}"
            return f"list\\[{type(obj[0]).__name__ if obj else 'empty'}\\] x {len(obj)}"
        else:
            return str(type(obj).__name__)

    def add_to_tree(tree, d):
        if isinstance(d, dict) or isinstance(d, BaseDataObject):
            for k, v in d.items():
                if isinstance(v, dict) or isinstance(v, BaseDataObject):
                    branch = tree.add(f"[bold]{k}[/bold] {describe(v)}")
                    add_to_tree(branch, v)
                else:
                    tree.add(f"[bold]{k}[/bold]: {describe(v)}")
        else:
            tree.add(describe(d))

    tree = Tree("[bold]data_dict[/bold]")
    add_to_tree(tree, data_dict)
    rprint(tree)


def get_config_info(cfg):
    overrides = HydraConfig.get().overrides.task
    defaults = cfg.defaults if "defaults" in cfg else []

    # 1. Build a mapping of override keys (from CLI or config)
    override_map = {}
    for item in overrides:
        if "=" in item:
            k, v = item.split("=", 1)
            override_map[k] = f"CLI ({item})"

    # 2. Traverse defaults chain to see where each group came from
    default_sources = {}
    for d in defaults:
        if isinstance(d, dict):
            for k, v in d.items():
                if v is not None:
                    default_sources[k] = f"{k}/{v}.yaml"
        elif isinstance(d, str):
            name = d.split("/")[0]
            default_sources[name] = f"{d}.yaml"

    # 3. Traverse config tree
    table_data = []

    def recurse(node, prefix=""):
        if not OmegaConf.is_dict(node):
            return
        for k, v in node.items():
            key_path = f"{prefix}.{k}" if prefix else k
            if OmegaConf.is_dict(v):
                recurse(v, prefix=key_path)
            else:
                # Who set it?
                override = override_map.get(key_path, default_sources.get(k, ""))
                source = getattr(cfg._metadata.config_sources.get(k, {}), "path", "")
                table_data.append(
                    (key_path, str(v), override or "default", source or "")
                )

    recurse(cfg)
    return table_data


def display_config_sources(cfg):
    data = get_config_info(cfg)
    table = Table(title="Hydra Config Resolution")
    table.add_column("Key", style="cyan")
    table.add_column("Final Value", style="magenta")
    table.add_column("Overridden By", style="green")
    table.add_column("Source Config", style="yellow")

    for row in data:
        table.add_row(*row)

    Console(width=180).print(table)


def print_transforms_summary(summary: dict[str, str]):
    table = Table(title="Data Transforms Summary")
    table.add_column("Transform Name", style="cyan", no_wrap=True)
    table.add_column("Time (s)", style="magenta")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green", overflow="fold")
    table.add_column("Keys Pre", style="yellow", overflow="fold")
    table.add_column("Keys Post", style="yellow", overflow="fold")

    for summary_dict in summary:
        table.add_row(
            summary_dict["transform_name"],
            summary_dict["time"],
            summary_dict["status"],
            summary_dict["details"],
            summary_dict.get("keys_pre", ""),
            summary_dict.get("keys_post", ""),
        )

    Console(width=180).print(table)
