import torch.nn as nn
from typing import List, Dict, Any


def _populate_summary(
    module: nn.Module, name: str, depth: int, summary_list: List[Dict[str, Any]]
):
    """
    Populate a hierarchical summary with module parameter counts.

    Parameters
    ----------
    module : nn.Module
        Module to inspect.
    name : str
        Display name for the current module.
    depth : int
        Nesting depth used for indentation.
    summary_list : list[dict[str, Any]]
        Accumulator that stores the summary rows.
    """
    children = list(module.named_children())
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

    # Add modules that have parameters or are containers of other modules
    if not children and total_params == 0:
        return

    summary_list.append(
        {
            "name": name,
            "depth": depth,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
    )

    # --- KEY IMPROVEMENT HERE ---
    # Determine if the current module is a Sequential block to format child names
    is_sequential = isinstance(module, nn.Sequential)

    # Recurse into children
    for child_name, child_module in children:
        # If the parent module is Sequential, use a more descriptive name for the child
        display_name = (
            f"{child_module.__class__.__name__} ({child_name})"
            if is_sequential
            else child_name
        )

        _populate_summary(child_module, display_name, depth + 1, summary_list)

    # Handle parameters defined directly on the module (not in a child)
    direct_params = sum(p.numel() for p in module.parameters(recurse=False))
    if children and direct_params > 0:
        direct_trainable_params = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )
        summary_list.append(
            {
                "name": "(Direct Parameters)",
                "depth": depth + 1,
                "total_params": direct_params,
                "trainable_params": direct_trainable_params,
            }
        )


def get_model_summary(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Generate a hierarchical parameter summary for a model.

    Parameters
    ----------
    model : nn.Module
        Model to summarize.

    Returns
    -------
    list[dict[str, Any]]
        Summary rows describing each module and the total parameter count.
    """
    summary_list = []
    for name, module in model.named_children():
        _populate_summary(module, name, 0, summary_list)

    summary_list.append(
        {
            "name": "Total",
            "depth": -1,
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
    )
    return summary_list


def print_model_summary(summary_list: List[Dict[str, Any]]):
    """
    Print a formatted hierarchical summary table.

    Parameters
    ----------
    summary_list : list[dict[str, Any]]
        Summary rows produced by :func:`get_model_summary`.
    """
    name_width = 45
    total_width = 20
    trainable_width = 25

    header = f"{'Module':<{name_width}} | {'Total Parameters':>{total_width}} | {'Trainable Parameters':>{trainable_width}}"
    print("-" * len(header))
    print(header)
    print("=" * len(header))

    for item in summary_list:
        if item["depth"] == -1:
            print("-" * len(header))
            name = item["name"]
            indent = ""
        else:
            name = item["name"]
            indent = "  " * item["depth"]

        total_str = f"{item['total_params']:,}"
        trainable_str = f"{item['trainable_params']:,}"
        display_name = f"{indent}{name}"

        print(
            f"{display_name:<{name_width}} | {total_str:>{total_width}} | {trainable_str:>{trainable_width}}"
        )
    print()
