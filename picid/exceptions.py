"""Project-wide exception types."""

from __future__ import annotations

from typing import Any


def build_transform_error(context: Any, e: BaseException) -> TransformError:
    """Build a TransformError from a pipeline context and the caught exception."""
    ti = getattr(context, "transform_instance", None)
    step_id = getattr(context, "step_id", None) or (
        getattr(ti, "__class__", None) and getattr(ti.__class__, "__name__", None)
    )
    transform_class = getattr(ti, "__class__", None) and getattr(
        ti.__class__, "__name__", None
    )
    apply_to = getattr(context, "apply_to_keys", None)
    msg = (
        "Transform failed. "
        f"Transform (config name): {step_id!r}. "
        f"Transform class: {transform_class or 'unknown'}. "
        f"Apply-to keys: {apply_to!r}. "
        f"Original error: {e!s}"
    )
    return TransformError(
        msg,
        step_id=step_id,
        transform_class=transform_class,
        apply_to_keys=apply_to,
        cause=e,
    )


class TransformError(Exception):
    """Raised when a data transform fails, with context about which transform and why."""

    def __init__(
        self,
        message: str,
        *,
        step_id: str | None = None,
        transform_class: str | None = None,
        apply_to_keys: list | None = None,
        cause: BaseException | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.step_id = step_id
        self.transform_class = transform_class
        self.apply_to_keys = apply_to_keys
        self.cause = cause

    def __str__(self) -> str:
        lines = ["Transform failed.", ""]
        if self.step_id is not None:
            lines.append(f"  Transform (config name): {self.step_id!r}")
        if self.transform_class is not None:
            lines.append(f"  Transform class: {self.transform_class}")
        if self.apply_to_keys:
            lines.append(f"  Apply-to keys: {self.apply_to_keys}")
        if (
            self.step_id is not None
            or self.transform_class is not None
            or self.apply_to_keys
        ):
            lines.append("")
        if self.cause is not None:
            lines.append("Original error:")
            # Indent the cause so it stands out; support multi-line cause messages
            cause_str = str(self.cause).strip()
            for part in cause_str.split("\n"):
                lines.append(f"  {part}")
        else:
            lines.append(self.message)
        return "\n".join(lines)
