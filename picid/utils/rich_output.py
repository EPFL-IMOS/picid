import logging
from pathlib import Path
from typing import Any

import numpy as np
import awkward as ak
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree
from rich.text import Text

from picid.data.data_objects import BaseDataObject
from picid.utils.awkward_utils import get_ak_shape, ak_find_var_dims

logger = logging.getLogger(__name__)


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


def print_hydra_config_tree(
    cfg: DictConfig, resolve: bool = True, title: str = "HYDRA CONFIG"
) -> Tree:
    """Build a Rich tree for the final Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Fully composed Hydra config.
    resolve : bool, default=True
        Whether to resolve interpolations before rendering.
    title : str, default="HYDRA CONFIG"
        Root label shown at the top of the tree.

    Returns
    -------
    Tree
        A Rich tree that renders the complete config hierarchy.
    """

    tree = Tree(title, style="dim", guide_style="dim")
    container = _to_config_tree_container(cfg, resolve=resolve)

    def add_node(parent: Tree, key: str, value: Any) -> None:
        if isinstance(value, dict):
            branch = parent.add(Text(key, style="bold"))
            if not value:
                branch.add("{}")
                return
            for child_key, child_value in value.items():
                add_node(branch, str(child_key), child_value)
        elif isinstance(value, list):
            branch = parent.add(Text(key, style="bold"))
            if not value:
                branch.add("[]")
                return
            for index, child_value in enumerate(value):
                add_node(branch, f"[{index}]", child_value)
        else:
            label = Text()
            label.append(key, style="bold")
            label.append(": ")
            label.append(str(value))
            parent.add(label)

    if isinstance(container, dict):
        for key, value in container.items():
            add_node(tree, str(key), value)
    elif isinstance(container, list):
        for index, value in enumerate(container):
            add_node(tree, f"[{index}]", value)
    else:
        tree.add(str(container))

    return tree


def _to_config_tree_container(cfg: DictConfig, resolve: bool) -> Any:
    if not resolve:
        return OmegaConf.to_container(cfg, resolve=False)

    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except OmegaConfBaseException as exc:
        container = _to_resolved_container_without_uninitialized_hydra(cfg)
        if container is not None:
            logger.warning(
                "Could not fully resolve Hydra config tree because Hydra runtime "
                "metadata is not initialized; rendering config without the top-level "
                "hydra section. Original error: %s",
                exc,
            )
            return container

        logger.warning(
            "Could not fully resolve Hydra config tree; rendering unresolved config. "
            "Original error: %s",
            exc,
        )
        return OmegaConf.to_container(cfg, resolve=False)


def _to_resolved_container_without_uninitialized_hydra(cfg: DictConfig) -> Any | None:
    if HydraConfig.initialized() or not isinstance(cfg, DictConfig) or "hydra" not in cfg:
        return None

    try:
        unresolved_container = OmegaConf.to_container(cfg, resolve=False)
        if not isinstance(unresolved_container, dict) or "hydra" not in unresolved_container:
            return None

        display_container = dict(unresolved_container)
        display_container.pop("hydra", None)
        display_cfg = OmegaConf.create(display_container)
        return OmegaConf.to_container(display_cfg, resolve=True)
    except OmegaConfBaseException:
        return None


def _describe_list_of_ak_arrays(obj: list[ak.Array], calculate_stat=False) -> str:
    """
    Analyze a list of Awkward Arrays and format compact shape statistics.

    Parameters
    ----------
    obj : list[ak.Array]
        List of arrays to describe.
    calculate_stat : bool, default=False
        Whether to compute per-dimension min/max/mean statistics.

    Returns
    -------
    str
        Human-readable description of the list contents.
    """
    if not obj:
        return "list<ak.Array> x 0"

    try:
        if calculate_stat:
            # --- (Data calculation part is unchanged) ---
            first_arr = obj[0]
            num_dims = first_arr.ndim
            var_dims = ak_find_var_dims(first_arr)
            min_vals, max_vals, mean_vals = ([0.0] * num_dims for _ in range(3))
            dim_labels = [str(i) for i in range(num_dims)]

            ak_shapes = [get_ak_shape(arr) for arr in obj]

            for i in range(num_dims):
                if i in var_dims:
                    # STEP 1: Get a simple list of lengths in one pass.
                    # This is much faster than converting each item to a NumPy array and concatenating.
                    lengths = [ak.num(arr, axis=i) for arr in obj]

                    # STEP 2: Calculate stats if the list is not empty.
                    if lengths:
                        lengths_arr = ak.Array(lengths)
                        min_vals[i] = ak.min(lengths_arr).item()
                        max_vals[i] = ak.max(lengths_arr).item()
                        mean_vals[i] = ak.mean(lengths_arr).item()

                    dim_labels[i] = f"{i} (var)"
                else:
                    lengths = list(map(lambda x: x[i], ak_shapes))
                    min_vals[i] = np.min(lengths)
                    max_vals[i] = np.max(lengths)
                    mean_vals[i] = np.mean(lengths)

        output_lines = [
            f"list<ak.Array> x {len(obj)} shape{get_ak_shape(obj[0])} (min/max/mean):"
        ]
        if calculate_stat:
            for i in range(num_dims):
                is_last_dim = i == num_dims - 1
                dim_prefix = "└── " if is_last_dim else "├── "

                # Add the line for the dimension
                stats_str = (
                    f"{min_vals[i]:.0f} / {max_vals[i]:.0f} / {mean_vals[i]:.1f}"
                )
                output_lines.append(f"\n{dim_prefix}dim {dim_labels[i]}: {stats_str}")

        return "".join(output_lines)

    except Exception as e:
        return f"list<ak.Array> x {len(obj)} (stats failed: {e})"


def describe_data_type(obj, calculate_stat=False) -> str:
    """
    Describe a single data object.

    Parameters
    ----------
    obj : Any
        Object to describe.
    calculate_stat : bool, default=False
        Whether to compute list statistics when applicable.

    Returns
    -------
    str
        Human-readable description of the object.
    """
    if isinstance(obj, dict):
        return "{...}"
    elif isinstance(obj, np.ndarray):
        return f"np.ndarray{obj.shape}"
    elif torch.is_tensor(obj):
        return f"torch.Tensor{tuple(obj.shape)}"
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return f"{type(obj).__name__}{obj.shape}"
    elif isinstance(obj, ak.Array):
        return f"ak.Array {str(obj.type)}"

    elif isinstance(obj, list):
        if not obj:
            return "list<empty> x 0"

        first = obj[0]
        if isinstance(first, ak.Array):
            # Route to the simplified helper for awkward arrays
            return _describe_list_of_ak_arrays(obj, calculate_stat)
        else:
            # Fallback for other list types
            descr = type(first).__name__
            return (
                f"list<{descr}> x {len(obj)} [{describe_data_type(first, False)}, ...]"
            )

    else:
        return str(type(obj).__name__)


def to_descriptive_dict(d, calculate_stat=True):
    """
    Convert nested data into a descriptive dictionary of shapes and types.

    Parameters
    ----------
    d : Any
        Object or mapping to describe.
    calculate_stat : bool, default=True
        Whether to include statistics for list-like arrays.

    Returns
    -------
    dict
        Nested dictionary of shape/type descriptions.
    """
    if isinstance(d, dict) or isinstance(d, BaseDataObject):
        return {k: to_descriptive_dict(v) for k, v in d.items()}
    else:
        return describe_data_type(d, calculate_stat)


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
def print_data_dict_structure(data_dict, calculate_stat=True):
    def add_to_tree(tree, d):
        if isinstance(d, dict) or isinstance(d, BaseDataObject):
            for k, v in d.items():
                if isinstance(v, dict) or isinstance(v, BaseDataObject):
                    branch = tree.add(
                        f"[bold]{k}[/bold] {describe_data_type(v, calculate_stat)}"
                    )
                    add_to_tree(branch, v)
                else:
                    tree.add(
                        f"[bold]{k}[/bold]: {describe_data_type(v, calculate_stat)}"
                    )
        else:
            tree.add(describe_data_type(d, calculate_stat))

    tree = Tree("[bold]data_dict[/bold]")
    add_to_tree(tree, data_dict)
    return tree


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
    table.add_column("Inputs", style="yellow", overflow="fold")

    for summary_dict in summary:
        table.add_row(
            summary_dict["transform_name"],
            summary_dict["time"],
            summary_dict["status"],
            summary_dict["details"],
            summary_dict.get("changes", ""),
            summary_dict.get("added", ""),
            summary_dict.get("removed", ""),
            summary_dict.get("inputs", ""),
        )

    Console(width=180).print(table)


def transform_log_to_summary_string(transform_log: Any) -> str:
    """
    Extract a human-readable summary from a transform log payload.

    Transform strategies return per-split dictionaries. Prefer the training
    split for the table because it is the split used to fit stateful transforms.
    """
    if not transform_log:
        return "No additional transform log."
    if not isinstance(transform_log, dict):
        return str(transform_log)
    for preferred in ("train", "test"):
        if preferred in transform_log:
            return str(transform_log[preferred])
    first_key = next(iter(transform_log))
    logger.debug(
        "transform_log has no 'train' or 'test' key; using %r for summary.",
        first_key,
    )
    return str(transform_log[first_key])


def build_transform_error_renderables(
    transform_name: str,
    flags: dict,
    metadata: dict,
    first_segment_keys: list | None,
    first_segment_rows: list[
        tuple[str, str, str]
    ],  # (key, type_name, describe_data_type)
    case_analysis_line: str,
) -> list:
    """
    Build Rich renderables for transform error diagnostics.

    Parameters
    ----------
    transform_name : str
        Name of the failing transform.
    flags : dict
        Data-type flags used in the diagnostic table.
    metadata : dict
        Metadata payload to display.
    first_segment_keys : list | None
        Optional keys for the first segment.
    first_segment_rows : list[tuple[str, str, str]]
        Rows describing the first segment contents.
    case_analysis_line : str
        Human-readable case-analysis summary.

    Returns
    -------
    list
        Rich renderables describing the failure.
    """
    out = []
    out.append(
        Rule(
            title=f"Transform Error — {transform_name}.transform_multi_source()",
            style="red",
        )
    )
    flags_table = Table(title="Data type flags")
    flags_table.add_column("Flag", style="cyan")
    flags_table.add_column("Value", style="yellow")
    for k, v in flags.items():
        flags_table.add_row(k, str(v))
    out.append(flags_table)
    out.append(Panel(str(metadata), title="Metadata", border_style="dim"))
    if first_segment_keys is not None and first_segment_rows:
        seg_table = Table(title="First segment contents")
        seg_table.add_column("key", style="cyan", no_wrap=True)
        seg_table.add_column("type", style="magenta")
        seg_table.add_column("description", style="green", overflow="fold")
        for row in first_segment_rows:
            seg_table.add_row(*row)
        out.append(seg_table)
    out.append(Panel(case_analysis_line, title="Case analysis", border_style="blue"))
    out.append(Rule(style="red"))
    return out
