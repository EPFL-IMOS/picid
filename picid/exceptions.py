"""Project-wide exception types."""

from __future__ import annotations

from typing import Any

from picid.data.datasources.base.exceptions import DatasourceError


def build_transform_error(context: Any, e: BaseException) -> TransformError:
    """
    Build a TransformError from a pipeline context and the caught exception.

    Parameters
    ----------
    context : Any
        Pipeline context that carries transform metadata.
    e : BaseException
        Original exception raised by the transform.

    Returns
    -------
    TransformError
        Exception enriched with transform identity and original cause.
    """
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


def build_preprocessing_datasource_error(
    stage: str, datasource: Any, e: DatasourceError
) -> PreprocessingDatasourceError:
    """
    Build a preprocessing wrapper around a datasource-layer exception.

    Parameters
    ----------
    stage : str
        Datasource operation that failed, for example ``"load_data"`` or
        ``"get_data"``.
    datasource : Any
        Datasource instance being used by the preprocessing pipeline.
    e : DatasourceError
        Original datasource-layer exception.

    Returns
    -------
    PreprocessingDatasourceError
        Exception enriched with datasource identity and stage context.
    """

    datasource_type = type(datasource).__name__ if datasource is not None else None
    datasource_name = _resolve_datasource_name(datasource)
    message = (
        "Datasource step failed during preprocessing. "
        f"Stage: {stage!r}. "
        f"Datasource: {datasource_type or 'unknown'}."
    )
    return PreprocessingDatasourceError(
        message,
        stage=stage,
        datasource_type=datasource_type,
        datasource_name=datasource_name,
        cause=e,
    )


def _resolve_datasource_name(datasource: Any) -> str | list[str] | None:
    """
    Resolve a human-readable datasource name for error reporting.

    Parameters
    ----------
    datasource : Any
        Datasource-like object used by preprocessing.

    Returns
    -------
    str | list[str] | None
        Best-effort datasource identifier, or ``None`` when unavailable.
    """

    if datasource is None:
        return None

    value = getattr(datasource, "data_name", None)
    if isinstance(value, str):
        return value

    getter = getattr(datasource, "get_data_names", None)
    if callable(getter):
        try:
            names = getter()
        except Exception:
            names = None
        if isinstance(names, tuple) and len(names) == 1 and isinstance(names[0], str):
            return names[0]
        if isinstance(names, tuple) and all(isinstance(name, str) for name in names):
            return list(names)

    getter = getattr(datasource, "get_data_name", None)
    if callable(getter):
        try:
            name = getter()
        except Exception:
            return None
        if isinstance(name, str):
            return name
        if isinstance(name, (list, tuple)) and all(
            isinstance(part, str) for part in name
        ):
            return list(name)
    return None


class TransformError(Exception):
    """
    Raised when a data transform fails.

    Parameters
    ----------
    message : str
        Human-readable error summary.
    step_id : str | None, optional
        Transform config name or step identifier.
    transform_class : str | None, optional
        Concrete transform class name.
    apply_to_keys : list | None, optional
        Keys that the transform was applied to.
    cause : BaseException | None, optional
        Original exception raised by the transform.
    """

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


class PreprocessingDatasourceError(Exception):
    """
    Raised when preprocessing fails while interacting with a datasource.

    Parameters
    ----------
    message : str
        Summary describing the preprocessing datasource failure.
    stage : str | None, optional
        Datasource operation that failed within preprocessing.
    datasource_type : str | None, optional
        Concrete datasource class name.
    datasource_name : str | list[str] | None, optional
        Best-effort logical datasource identifier.
    cause : BaseException | None, optional
        Original datasource-layer exception.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        datasource_type: str | None = None,
        datasource_name: str | list[str] | None = None,
        cause: BaseException | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.datasource_type = datasource_type
        self.datasource_name = datasource_name
        self.cause = cause
        self.datasource_error_type = type(cause).__name__ if cause is not None else None

    def __str__(self) -> str:
        lines = ["Datasource failed during preprocessing.", ""]
        if self.stage is not None:
            lines.append(f"  Stage: {self.stage!r}")
        if self.datasource_type is not None:
            lines.append(f"  Datasource class: {self.datasource_type}")
        if self.datasource_name is not None:
            lines.append(f"  Datasource name: {self.datasource_name!r}")
        if (
            self.stage is not None
            or self.datasource_type is not None
            or self.datasource_name is not None
        ):
            lines.append("")
        if self.cause is not None:
            lines.append("Original datasource error:")
            cause_str = str(self.cause).strip()
            prefix = self.datasource_error_type or type(self.cause).__name__
            cause_lines = cause_str.split("\n") if cause_str else [repr(self.cause)]
            if cause_lines:
                lines.append(f"  {prefix}: {cause_lines[0]}")
                for part in cause_lines[1:]:
                    lines.append(f"  {part}")
        else:
            lines.append(self.message)
        return "\n".join(lines)
