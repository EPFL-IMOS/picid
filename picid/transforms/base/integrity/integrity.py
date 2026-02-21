"""
Integrity guards — policy-driven invariants and validators.

IntegrityPolicy controls whether a check violation fails, warns, or is allowed.
Checks can be injected into the pipeline via IntegrityCheckStep (optional).
NaN check remains available as a built-in check; mandatory-by-default is achieved
by using policy "fail" when the check is registered.
"""
from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple

IntegrityPolicy = str  # "fail" | "warn" | "allow"
CHECK_FN = Callable[[], None]  # no-arg callable; may raise


def run_check(
    name: str,
    check_fn: CHECK_FN,
    policy: IntegrityPolicy,
) -> None:
    """
    Run a single integrity check; apply policy on violation (when check_fn raises).

    - fail: re-raise the exception.
    - warn: issue a warning with the exception message, then continue.
    - allow: suppress the exception and continue.
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
    """Run a list of (name, check_fn, policy) in order. Stops on first 'fail' raise."""
    for name, check_fn, policy in checks:
        run_check(name, check_fn, policy)
