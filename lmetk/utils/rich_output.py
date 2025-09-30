from pathlib import Path

import numpy as np
import awkward as ak
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from lmetk.data.data_objects.base import BaseDataObject
from lmetk.utils.awkward_utils import get_ak_shape, ak_find_var_dims


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


import numpy as np
import torch
import pandas as pd
import awkward as ak
from typing import List, Union

# Assume your provided utility functions are in this file or imported
# ak_find_var_dims, get_ak_shape, etc.


def _describe_list_of_ak_arrays(obj: list[ak.Array]) -> str:
    """
    Analyzes a list of awkward arrays and formats the stats in a compact,
    single-line-per-dimension tree.
    """
    if not obj:
        return "list<ak.Array> x 0"

    try:
        # --- (Data calculation part is unchanged) ---
        first_arr = obj[0]
        num_dims = first_arr.ndim
        var_dims = ak_find_var_dims(first_arr)
        min_vals, max_vals, mean_vals = ([0.0] * num_dims for _ in range(3))
        dim_labels = [str(i) for i in range(num_dims)]

        for i in range(num_dims):
            if i in var_dims:
                all_lengths = np.concatenate(
                    [ak.to_numpy(ak.num(arr, axis=i)) for arr in obj]
                )
                if all_lengths.size > 0:
                    min_vals[i] = np.min(all_lengths)
                    max_vals[i] = np.max(all_lengths)
                    mean_vals[i] = np.mean(all_lengths)
                dim_labels[i] = f"{i} (var)"
            else:
                lengths = [get_ak_shape(arr)[i] for arr in obj]
                min_vals[i] = np.min(lengths)
                max_vals[i] = np.max(lengths)
                mean_vals[i] = np.mean(lengths)

        # --- (NEW compact formatting logic) ---
        output_lines = [
            f"list<ak.Array> x {len(obj)} shape:{get_ak_shape(obj)} (min/max/mean)"
        ]

        for i in range(num_dims):
            is_last_dim = i == num_dims - 1
            dim_prefix = "└── " if is_last_dim else "├── "

            # Add the line for the dimension
            stats_str = f"{min_vals[i]:.0f} / {max_vals[i]:.0f} / {mean_vals[i]:.1f}"
            output_lines.append(f"\n{dim_prefix}dim {dim_labels[i]}: {stats_str}")

        return "".join(output_lines)

    except Exception as e:
        return f"list<ak.Array> x {len(obj)} (stats failed: {e})"


def describe_data_type(obj) -> str:
    """Describes a single data object."""
    if isinstance(obj, dict):
        return "{...}"
    elif isinstance(obj, np.ndarray):
        return f"np.ndarray{obj.shape}"
    elif torch.is_tensor(obj):
        return f"torch.Tensor{tuple(obj.shape)}"
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return f"{type(obj).__name__}{obj.shape}"

    elif isinstance(obj, list):
        if not obj:
            return "list<empty> x 0"

        first = obj[0]
        if isinstance(first, ak.Array):
            # Route to the simplified helper for awkward arrays
            return _describe_list_of_ak_arrays(obj)
        else:
            # Fallback for other list types
            descr = type(first).__name__
            return f"list<{descr}> x {len(obj)}"

    else:
        return str(type(obj).__name__)


# def describe_data_type(obj):
#     if isinstance(obj, dict) or isinstance(obj, BaseDataObject):
#         return "{...}"
#     elif isinstance(obj, np.ndarray):
#         return f"np.ndarray{obj.shape}"
#     elif torch.is_tensor(obj):
#         return f"torch.Tensor{tuple(obj.shape)}"
#     elif isinstance(obj, pd.DataFrame):
#         return f"pd.DataFrame{obj.shape}"
#     elif isinstance(obj, pd.Series):
#         return f"pd.Series{obj.shape}"
#     elif isinstance(obj, ak.Array):
#         return f"ak.Array {get_ak_shape(obj)}"
#     elif isinstance(obj, list):
#         if obj and (
#             torch.is_tensor(obj[0])
#             or isinstance(obj[0], np.ndarray)
#             or isinstance(obj[0], pd.DataFrame)
#             or isinstance(obj[0], pd.Series)
#             or isinstance(obj[0], ak.Array)
#         ):
#             descr = describe_data_type(obj[0])
#         else:
#             descr = type(obj[0]).__name__ if obj else "empty"
#         return f"list<{descr}> x {len(obj)}"
#     else:
#         return str(type(obj).__name__)


def to_descriptive_dict(d):
    """Convert a (possibly nested) dictionary to a descriptive dictionary showing data types and shapes."""
    if isinstance(d, dict) or isinstance(d, BaseDataObject):
        return {k: to_descriptive_dict(v) for k, v in d.items()}
    else:
        return describe_data_type(d)


def descriptive_dict_differences_str(old, new, mode="added"):

    assert mode in ("added", "removed", "changed"), "Invalid mode"

    """Print the differences in a flat, non-nested dictionary based on the specified mode."""
    differences = []
    all_keys = old.keys() | new.keys()

    for key in all_keys:
        if mode == "added" and key not in old:
            differences.append(f"{key}: {new[key]}")
        elif mode == "removed" and key not in new:
            differences.append(f"{key}: {old[key]}")
        elif mode == "changed" and key in old and key in new:
            val1 = old[key]
            val2 = new[key]
            if val1 != val2:
                differences.append(f"{key}: {val1} -> {val2}")

    return "\n".join(differences)


# Plot the data_dict structure with tensor dimensions
def print_data_dict_structure(data_dict):

    def add_to_tree(tree, d):
        if isinstance(d, dict) or isinstance(d, BaseDataObject):
            for k, v in d.items():
                if isinstance(v, dict) or isinstance(v, BaseDataObject):
                    branch = tree.add(f"[bold]{k}[/bold] {describe_data_type(v)}")
                    add_to_tree(branch, v)
                else:
                    tree.add(f"[bold]{k}[/bold]: {describe_data_type(v)}")
        else:
            tree.add(describe_data_type(d))

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
    table.add_column("Changes", style="yellow", overflow="fold")
    table.add_column("Added", style="yellow", overflow="fold")
    table.add_column("Removed", style="yellow", overflow="fold")

    for summary_dict in summary:
        table.add_row(
            summary_dict["transform_name"],
            summary_dict["time"],
            summary_dict["status"],
            summary_dict["details"],
            summary_dict.get("changes", ""),
            summary_dict.get("added", ""),
            summary_dict.get("removed", ""),
        )

    Console(width=180).print(table)
