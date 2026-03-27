"""
Integrity check execution and default before/after checks.

Used by IntegrityCheckBeforeStep and IntegrityCheckAfterStep (in steps.py):
before-step runs checks on the chunk dict per split; after-step runs checks on
the assign_to slice of transformed data. Check tuples are (name, fn, policy);
policy "fail" raises on exception, "warn" logs and continues.

Default checks: before = no NaNs + length consistency; after = structure +
split preservation. Built by build_default_before_checks / build_default_after_checks
and attached to TransformContext by TransformStrategy.
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
    Run (name, fn, policy) checks, passing data_slice as the sole argument.

    - policy "fail": on exception, raise RuntimeError (message includes step_id/split/stage if given).
    - policy "warn": on exception, log and continue; name still appended to ran.
    - policy "allow": exceptions are not specially handled (same as warn for now).

    step_id, split, and stage are optional; when provided they are included in the
    RuntimeError message so pipeline failures can be traced to a specific
    transform, split, and pipeline stage (e.g. "before-transform", "after-transform").
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
    """Default before-check: no NaN values in any chunk of this split."""
    from picid.data.data_objects.utils import check_for_nans

    for split, chunks in split_chunks.items():
        for chunk in chunks:
            check_for_nans(
                list(chunk.values()), list(chunk.keys()), raise_on_error=True
            )


def _default_before_check_lengths(
    split_chunks: Dict[str, List[NamedTransformInput]],
) -> None:
    """Default before-check: consistent array lengths within each chunk."""
    from picid.data.data_objects.utils import check_length_consistency

    for split, chunks in split_chunks.items():
        for chunk in chunks:
            check_length_consistency(
                list(chunk.values()), list(chunk.keys()), raise_on_error=True
            )


def _default_after_check_structure(
    assign_to_slice: Dict[str, Any],
) -> None:
    """Default after-check: every assign_to key maps to a split-keyed dict."""
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
    """Default after-check: split set is unchanged after transformation."""
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
    Return the default before-checks run on each split's chunks.

    Checks: no NaN in any chunk value; consistent array lengths per chunk.
    Used when validate_output=True in the strategy (default).
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
    Return the default after-checks run on the assign_to slice.

    Checks: each assign_to key maps to a split-keyed mapping; split set is
    unchanged from original_splits. Used when validate_output=True (default).
    """

    def _check_splits(assign_to_slice: Dict[str, Any]) -> None:
        _default_after_check_splits(assign_to_slice, original_splits)

    return [
        ("check_output_structure", _default_after_check_structure, policy),
        ("check_split_preservation", _check_splits, policy),
    ]
