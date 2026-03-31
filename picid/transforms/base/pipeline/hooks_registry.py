"""
Pipeline lifecycle hook registry and emission.

The pipeline (orchestrated by TransformStrategy) emits events at key points:
on_pipeline_start, before_transform, after_transform, on_transform_error,
on_pipeline_end. Callbacks registered here receive (event_name, TransformContext)
and can be used for logging, metrics, or debugging.

Default hooks (e.g. logging, timing) are registered lazily on first _emit() to
avoid circular imports with the base hooks module.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from picid.transforms.base.pipeline.context import TransformContext

HookCallback = Callable[[str, TransformContext], None]
_HOOK_REGISTRY: Dict[str, List[HookCallback]] = {
    "on_pipeline_start": [],
    "before_transform": [],
    "after_transform": [],
    "on_transform_error": [],
    "on_pipeline_end": [],
}

_default_hooks_registered = False


def register_hook(event: str, callback: HookCallback) -> None:
    """
    Register a callback for the given pipeline event.

    Parameters
    ----------
    event : str
        Lifecycle event name.
    callback : HookCallback
        Callback invoked with ``(event, context)``.
    """
    if event not in _HOOK_REGISTRY:
        _HOOK_REGISTRY[event] = []
    _HOOK_REGISTRY[event].append(callback)


def clear_hooks(event: Optional[str] = None) -> None:
    """
    Remove callbacks for one event or for all events.

    Parameters
    ----------
    event : str, optional
        Event name to clear. When omitted, all hooks are removed.
    """
    if event is None:
        for k in _HOOK_REGISTRY:
            _HOOK_REGISTRY[k] = []
    elif event in _HOOK_REGISTRY:
        _HOOK_REGISTRY[event] = []


def _register_default_hooks() -> None:
    """Import and register the default hooks from ``picid.transforms.base``."""
    from picid.transforms.base import hooks

    hooks.register_defaults()


def _emit(event: str, context: TransformContext) -> None:
    """
    Emit a pipeline lifecycle event to all registered callbacks.

    Parameters
    ----------
    event : str
        Lifecycle event name.
    context : TransformContext
        Shared pipeline context.

    Notes
    -----
    Default hooks are registered on first call to avoid circular imports.
    Exceptions in callbacks are swallowed so hooks never break the pipeline.
    """
    global _default_hooks_registered
    if not _default_hooks_registered:
        _default_hooks_registered = True
        _register_default_hooks()
    for cb in _HOOK_REGISTRY.get(event, []):
        try:
            cb(event, context)
        except Exception:
            pass  # hooks must never break the pipeline
