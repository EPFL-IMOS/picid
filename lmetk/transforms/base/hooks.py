"""
Default and optional hook implementations for the transform pipeline.

Hook implementations live here so pipeline.py stays focused on the registry,
steps, and emission logic. Register defaults via register_defaults().
"""
from __future__ import annotations

import logging

from lmetk.transforms.base.pipeline import TransformContext, register_hook


def on_transform_error_enrich_log(event: str, context: TransformContext) -> None:
    """Log pipeline failures with rich context for debugging and error tracking."""
    if context.error is None:
        return
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
    logging.getLogger("lmetk.transforms").error(
        "Pipeline failed: %s",
        context.error,
        exc_info=context.error,
        extra=extra,
    )


def register_defaults() -> None:
    """Register default pipeline hooks (e.g. enrich-and-re-log on transform error)."""
    register_hook("on_transform_error", on_transform_error_enrich_log)
