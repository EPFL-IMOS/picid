"""
Integrity check execution and default before/after checks.

This module provides the small, self-contained checks used by
``IntegrityCheckBeforeStep`` and ``IntegrityCheckAfterStep`` in
``steps.py``. Before checks run on the chunk dictionary for each split,
whereas after checks run on the ``assign_to`` slice of transformed data.
Each check tuple is ``(name, fn, policy)``; ``policy="fail"`` raises on
exception, ``policy="warn"`` logs and continues.

The default check sets are:

* before: no NaNs and length consistency
* after: output structure preservation and split-set preservation

They are built by :func:`build_default_before_checks` and
:func:`build_default_after_checks`, then attached to
``TransformContext`` by ``TransformStrategy``.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any, Callable, Dict, List, Tuple

from picid.data.data_objects import NamedTransformInput

from picid.transforms.base.pipeline.context import AfterCheck, BeforeCheck

logger = logging.getLogger(__name__)


def _run_scoped_checks(
    checks: List[Tuple[str, Callable, str]],
    data_slice: Any,
    *,
    step_id: str | None = None,
    split: str | None = None,
    stage: str | None = None,
) -> List[str]:
    """
    Run scoped integrity checks against a data slice.

    Parameters
    ----------
    checks : list[tuple[str, callable, str]]
        List of ``(name, fn, policy)`` tuples to execute.
    data_slice : Any
        The object passed as the sole argument to each check callable.
    step_id : str, optional
        Optional transform identifier used to enrich failure messages.
    split : str, optional
        Optional split name used to enrich failure messages.
    stage : str, optional
        Optional pipeline stage name used to enrich failure messages.

    Returns
    -------
    list[str]
        Names of the checks that ran successfully, or that emitted a warning.

    Notes
    -----
    ``policy="fail"`` raises a ``RuntimeError`` on exceptions. ``policy="warn"``
    logs the error and continues. ``policy="allow"`` is treated like ``warn``
    for now.
    """
    ran: List[str] = []
    for name, fn, policy in checks:
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) == 0:
                warnings.warn(
                    f"Integrity check '{name}' uses a no-arg signature. "
                    "Pass the data slice as the first argument. "
                    "No-arg checks will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=4,
                )
                fn()
            else:
                fn(data_slice)
            ran.append(name)
        except Exception as exc:
            if policy == "fail":
                prefix_parts = []
                if stage is not None:
                    prefix_parts.append(f"[{stage}]")
                if step_id is not None:
                    prefix_parts.append(f"transform {step_id!r}")
                if split is not None:
                    prefix_parts.append(f"split {split!r}")
                prefix = " ".join(prefix_parts)
                msg = (
                    f"{prefix}: integrity check {name!r} failed: {exc}"
                    if prefix
                    else f"Integrity check '{name}' failed: {exc}"
                )
                raise RuntimeError(msg) from exc
            elif policy == "warn":
                logger.warning("Integrity check '%s' failed (warn only): %s", name, exc)
                ran.append(name)
    return ran


def _default_before_check_nans(
    split_chunks: Dict[str, List[NamedTransformInput]],
) -> None:
    """
    Check that all chunks in a split contain no NaN values.

    Parameters
    ----------
    split_chunks : dict[str, list[NamedTransformInput]]
        Mapping from split name to the chunks belonging to that split.
    """
    from picid.data.data_objects.utils import check_for_nans

    for split, chunks in split_chunks.items():
        for chunk in chunks:
            check_for_nans(
                list(chunk.values()), list(chunk.keys()), raise_on_error=True
            )


def _default_before_check_lengths(
    split_chunks: Dict[str, List[NamedTransformInput]],
) -> None:
    """
    Check that each chunk contains arrays with consistent lengths.

    Parameters
    ----------
    split_chunks : dict[str, list[NamedTransformInput]]
        Mapping from split name to the chunks belonging to that split.
    """
    from picid.data.data_objects.utils import check_length_consistency

    for split, chunks in split_chunks.items():
        for chunk in chunks:
            check_length_consistency(
                list(chunk.values()), list(chunk.keys()), raise_on_error=True
            )


def _default_after_check_structure(
    assign_to_slice: Dict[str, Any],
) -> None:
    """
    Check that every ``assign_to`` key maps to a split-keyed mapping.

    Parameters
    ----------
    assign_to_slice : dict[str, Any]
        Mapping of output keys to their transformed split payload.
    """
    from collections.abc import Mapping

    for key, split_data in assign_to_slice.items():
        if not isinstance(split_data, Mapping):
            raise TypeError(
                f"Output key '{key}' must be a split-keyed mapping, "
                f"got {type(split_data).__name__}."
            )


def _default_after_check_splits(
    assign_to_slice: Dict[str, Any],
    original_splits: frozenset,
) -> None:
    """
    Check that the split set stays unchanged after transformation.

    Parameters
    ----------
    assign_to_slice : dict[str, Any]
        Mapping of output keys to their transformed split payload.
    original_splits : frozenset
        The split names expected to remain present after transformation.
    """
    from collections.abc import Mapping

    for key, split_data in assign_to_slice.items():
        if isinstance(split_data, Mapping):
            result_splits = frozenset(split_data.keys())
            if result_splits != original_splits:
                raise KeyError(
                    f"Output key '{key}': split set changed. "
                    f"Expected {set(original_splits)}, got {set(result_splits)}."
                )


def build_default_before_checks(policy: str = "fail") -> List[BeforeCheck]:
    """
    Build the default before-check list.

    Parameters
    ----------
    policy : str
        Error-handling policy assigned to each generated check tuple.

    Returns
    -------
    list[BeforeCheck]
        The default checks that run on each split's chunks.
    """
    return [
        ("check_no_nans", _default_before_check_nans, policy),
        ("check_length_consistency", _default_before_check_lengths, policy),
    ]


def build_default_after_checks(
    original_splits: frozenset,
    policy: str = "fail",
) -> List[AfterCheck]:
    """
    Build the default after-check list.

    Parameters
    ----------
    original_splits : frozenset
        Split names that must remain present after transformation.
    policy : str
        Error-handling policy assigned to each generated check tuple.

    Returns
    -------
    list[AfterCheck]
        The default checks that run on the transformed ``assign_to`` slice.
    """

    def _check_splits(assign_to_slice: Dict[str, Any]) -> None:
        _default_after_check_splits(assign_to_slice, original_splits)

    return [
        ("check_output_structure", _default_after_check_structure, policy),
        ("check_split_preservation", _check_splits, policy),
    ]
