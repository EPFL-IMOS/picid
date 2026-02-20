"""
Default and optional hook implementations for the transform pipeline.

Hook implementations live here so pipeline.py stays focused on the registry,
steps, and emission logic. Register defaults via register_defaults().
"""
from __future__ import annotations

import logging
import time

from rich.console import Console
from rich.table import Table

from lmetk.transforms.base.pipeline import TransformContext, register_hook


def set_transform_started_at(event: str, context: TransformContext) -> None:
    """Record pipeline start time for manifest timing (before_transform)."""
    context.started_at = time.time()


def set_transform_finished_at(event: str, context: TransformContext) -> None:
    """Record pipeline end time for manifest timing (after_transform)."""
    context.finished_at = time.time()


def on_transform_error_enrich_log(event: str, context: TransformContext) -> None:
    """Log pipeline failures with rich context for debugging and error tracking."""
    if context.error is None:
        return
    if context.finished_at is None:
        context.finished_at = time.time()
    extra = {
        "transform": getattr(
            context.transform_instance,
            "__name__",
            type(context.transform_instance).__name__,
        ),
        "apply_to_keys": context.apply_to_keys,
        "transform_on_keys": context.transform_on_keys,
        "available_splits": getattr(context, "available_splits", None),
        "fit_on_split": context.fit_on_split,
    }
    table = Table(title="Pipeline error context")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="yellow", overflow="fold")
    for k, v in extra.items():
        table.add_row(k, str(v))
    Console(stderr=True).print(table)
    logging.getLogger("lmetk.transforms").error(
        "Pipeline failed: %s",
        context.error,
        exc_info=True,
        extra=extra,
    )


def register_defaults() -> None:
    """Register default pipeline hooks (timing for manifest, enrich-and-re-log on error)."""
    register_hook("before_transform", set_transform_started_at)
    register_hook("after_transform", set_transform_finished_at)
    register_hook("on_transform_error", on_transform_error_enrich_log)
