"""
Default and optional hook implementations for the transform pipeline.

Hook implementations live here so pipeline.py stays focused on the registry,
steps, and emission logic. Register defaults via register_defaults().

on_transform_error hooks run in registration order: enrich first (so context.error
becomes a TransformError), then log (so table and log use the enriched error).
"""

from __future__ import annotations

import logging
import time

from rich.console import Console
from rich.table import Table

from picid.exceptions import TransformError, build_transform_error
from picid.transforms.base.pipeline import TransformContext, register_hook


def set_transform_started_at(event: str, context: TransformContext) -> None:
    """
    Record the pipeline start time for manifest timing.

    Parameters
    ----------
    event : str
        Hook event name.
    context : TransformContext
        Shared pipeline context.
    """
    context.started_at = time.time()


def set_transform_finished_at(event: str, context: TransformContext) -> None:
    """
    Record the pipeline end time for manifest timing.

    Parameters
    ----------
    event : str
        Hook event name.
    context : TransformContext
        Shared pipeline context.
    """
    context.finished_at = time.time()


def on_transform_error_enrich(event: str, context: TransformContext) -> None:
    """
    Replace ``context.error`` with a ``TransformError`` for later hooks.

    Parameters
    ----------
    event : str
        Hook event name.
    context : TransformContext
        Shared pipeline context.
    """
    if context.error is None:
        return
    if context.finished_at is None:
        context.finished_at = time.time()
    context.error = build_transform_error(context, context.error)


def on_transform_error_log(event: str, context: TransformContext) -> None:
    """
    Log pipeline failures with rich context.

    Parameters
    ----------
    event : str
        Hook event name.
    context : TransformContext
        Shared pipeline context.
    """
    if context.error is None:
        return
    err = context.error
    if isinstance(err, TransformError):
        extra = {
            "transform (config name)": err.step_id,
            "transform class": err.transform_class,
            "apply_to_keys": err.apply_to_keys,
            "available_splits": getattr(context, "available_splits", None),
            "fit_on_split": context.fit_on_split,
        }
    else:
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
    logging.getLogger("picid.transforms").error(
        "Pipeline failed: %s",
        err,
        exc_info=True,
        extra=extra,
    )


def register_defaults() -> None:
    """Register default pipeline hooks (timing for manifest; on error: enrich then log)."""
    register_hook("before_transform", set_transform_started_at)
    register_hook("after_transform", set_transform_finished_at)
    register_hook("on_transform_error", on_transform_error_enrich)
    register_hook("on_transform_error", on_transform_error_log)
