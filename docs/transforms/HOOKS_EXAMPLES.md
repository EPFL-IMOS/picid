# Transform pipeline hooks – example implementations

The pipeline emits lifecycle events that you can subscribe to via `register_hook(event, callback)`. Callbacks receive `(event: str, context: TransformContext)`. On **`on_transform_error`**, `context.error` is set to the exception that caused the pipeline to fail, so hooks can log or report it without changing the pipeline’s re-raise behaviour.

Events: `on_pipeline_start`, `before_transform`, `after_transform`, **`on_transform_error`**, `on_pipeline_end`.

On transform or pipeline errors, the default hooks print Rich tables and panels to stderr for diagnostics; the same events are also logged with traceback for log files.

Below are draft examples for **`on_transform_error`** to facilitate debugging (e.g. for the `transform_on_keys` empty-intersection case and other failures).

---

## 1. Debug logger (context + exception)

Log full context and exception so you can inspect state at failure time (e.g. in CI or locally):

```python
import logging
from picid.transforms.base.pipeline import register_hook

logger = logging.getLogger("picid.transforms")

def on_error_log(event: str, context) -> None:
    if context.error is None:
        return
    transform_name = getattr(
        context.transform_instance, "__name__",
        type(context.transform_instance).__name__ if context.transform_instance else None,
    )
    logger.error(
        "Transform pipeline error: transform=%s apply_to=%s transform_on_keys=%s available_splits=%s error=%s",
        transform_name,
        context.apply_to_keys,
        context.transform_on_keys,
        getattr(context, "available_splits", None),
        context.error,
        exc_info=context.error,
    )

register_hook("on_transform_error", on_error_log)
```

---

## 2. Pretty-print for `transform_on_keys` / CopyStep failures

When the failure is the “no splits left after applying transform_on_keys” case (or any `ValueError` in CopyStep), print a short, actionable summary:

```python
from picid.transforms.base.pipeline import register_hook

def on_error_transform_on_keys_hint(event: str, context) -> None:
    err = context.error
    if err is None or not isinstance(err, ValueError):
        return
    msg = str(err)
    if "no splits left after applying transform_on_keys" not in msg:
        return
    # Optional: log or print a one-liner that points to the fix
    apply_key = (context.apply_to_keys or [None])[0]
    data_splits = sorted(context.data[apply_key].keys()) if apply_key and apply_key in context.data else []
    print(
        f"[on_transform_error] transform_on_keys mismatch: "
        f"data has splits {data_splits}, requested {context.transform_on_keys}. "
        "Use one of the data splits in transform_on_keys or set transform_on_keys to None."
    )

register_hook("on_transform_error", on_error_transform_on_keys_hint)
```

---

## 3. Enrich and re-log (e.g. for error tracking)

You can attach context to your logging or error-tracking system without re-raising (the pipeline still re-raises after hooks run):

```python
import logging
from picid.transforms.base.pipeline import register_hook

def on_error_rich_context(event: str, context) -> None:
    if context.error is None:
        return
    extra = {
        "transform": getattr(context.transform_instance, "__name__", type(context.transform_instance).__name__),
        "apply_to_keys": context.apply_to_keys,
        "transform_on_keys": context.transform_on_keys,
        "available_splits": getattr(context, "available_splits", None),
        "fit_on_split": context.fit_on_split,
    }
    logging.getLogger("picid.transforms").error(
        "Pipeline failed: %s",
        context.error,
        exc_info=context.error,
        extra=extra,
    )
    # Optional: send to Sentry, etc., with extra as tags/breadcrumbs
    # sentry_sdk.set_context("transform", extra)
    # sentry_sdk.capture_exception(context.error)

register_hook("on_transform_error", on_error_rich_context)
```

---

## 4. Conditional behaviour by exception type

Run different logic for different errors (e.g. KeyError vs ValueError):

```python
from picid.transforms.base.pipeline import register_hook

def on_error_by_type(event: str, context) -> None:
    err = context.error
    if err is None:
        return
    if isinstance(err, ValueError) and "transform_on_keys" in str(err):
        # e.g. log to a dedicated channel or metric
        pass
    elif isinstance(err, KeyError):
        # e.g. “missing key” alert
        pass
    else:
        # generic handling
        pass

register_hook("on_transform_error", on_error_by_type)
```

---

## Usage note

- Register hooks once at app/experiment setup (e.g. in a `conftest.py` or main script).
- Use `clear_hooks("on_transform_error")` or `clear_hooks()` to remove them when needed (e.g. between tests).
- Hooks must not raise: the pipeline catches callback exceptions and continues so that the original error is still re-raised.
