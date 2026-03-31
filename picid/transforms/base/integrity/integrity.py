"""
Integrity guards for policy-driven invariants and validators.

IntegrityPolicy controls whether a check violation fails, warns, or is
allowed. Checks can be injected into the pipeline via an integrity check step.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Tuple

IntegrityPolicy = str  # "fail" | "warn" | "allow"
CHECK_FN = Callable[[], None]  # no-arg callable; may raise


def run_check(
    name: str,
    check_fn: CHECK_FN,
    policy: IntegrityPolicy,
) -> None:
    """
    Run a single integrity check; apply policy on violation (when check_fn raises).

    Parameters
    ----------
    name : str
        Human-readable check name used in warnings and errors.
    check_fn : CHECK_FN
        No-argument callable that performs the check.
    policy : IntegrityPolicy
        Handling policy for exceptions raised by ``check_fn``.

    Notes
    -----
    * ``fail`` re-raises the exception.
    * ``warn`` issues a warning with the exception message, then continues.
    * ``allow`` suppresses the exception and continues.
    """
    try:
        check_fn()
    except Exception as e:
        if policy == "fail":
            raise
        if policy == "warn":
            warnings.warn(
                f"Integrity check '{name}' failed (policy=warn): {e}",
                UserWarning,
                stacklevel=2,
            )
        # policy == "allow": do nothing


def run_checks(
    checks: List[Tuple[str, CHECK_FN, IntegrityPolicy]],
) -> None:
    """
    Run a list of checks in order.

    Parameters
    ----------
    checks : list[tuple[str, CHECK_FN, IntegrityPolicy]]
        Checks to execute in sequence.
    """
    for name, check_fn, policy in checks:
        run_check(name, check_fn, policy)
