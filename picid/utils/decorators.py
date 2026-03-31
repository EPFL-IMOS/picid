import functools
import logging
from typing import Any, Dict
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def inject_transform_context_to_strategy_apply(func):
    """
    Decorate transform strategy calls with better error context.

    The wrapper adds transform name and ``apply_to`` information to any
    exception raised by the wrapped function so debugging failures in
    transformation pipelines stays practical.

    Parameters
    ----------
    func : Callable
        Callable that implements the actual strategy method.

    Returns
    -------
    Callable
        Decorator that returns the wrapped strategy call.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Identify Context from Arguments (Expecting 'transform_instance' and 'apply_to_keys' in kwargs)
        transform_instance = kwargs.get("transform_instance")
        apply_to_keys = kwargs.get("apply_to_keys")

        # Best effort attempt to get context
        # Assumes the instance has a transform_name attribute or falls back to the class name
        t_name = getattr(
            transform_instance,
            "transform_name",
            (
                transform_instance.__class__.__name__
                if transform_instance
                else "Unknown Transform"
            ),
        )
        apply_to_str = (
            apply_to_keys if isinstance(apply_to_keys, str) else str(apply_to_keys)
        )

        try:
            return func(*args, **kwargs)

        except Exception as e:
            # 2. Construct the Context Message
            context_msg = (
                f"\n\n🚨 CRITICAL ERROR DURING TRANSFORMATION 🚨\n"
                f"TRANSFORM NAME: **{t_name}**\n"
                f"APPLY_TO KEY(s): **{apply_to_str}**\n"
                f"Original Exception: {e.__class__.__name__}: {e}\n"
            )

            # Log the original error with context
            logger.error(context_msg)

            # 3. Re-raise with the new context
            raise

    return wrapper


def check_transform_output_consistency(transform_func):
    """
    Validate that a transform output contains the expected mapping keys.

    The wrapper checks the returned mapping against the assignment metadata so
    the later merge step fails early with an actionable message when a required
    output key is missing.

    Parameters
    ----------
    transform_func : Callable
        Callable implementing the underlying transform method.

    Returns
    -------
    Callable
        Decorator that checks the wrapped transform output.
    """

    @functools.wraps(transform_func)
    def wrapper(self, data: Any, metadata: Dict[str, Any]) -> Any:
        # 1. Execute the actual transform logic to get the raw output
        raw_output = transform_func(self, data, metadata)

        # --- 2. Perform the Consistency Check ---

        assign_to_map = metadata.get("assign_to_map")
        expected_key = None

        # Check is only relevant if there is exactly one mapped assignment key
        if isinstance(assign_to_map, list) and len(assign_to_map) == 1:
            expected_key = assign_to_map[0]

        # We only check if the output is a mapping (dict-like) AND we have an explicit key to check
        if expected_key is not None and isinstance(raw_output, Mapping):
            if expected_key not in raw_output:
                # --- Construct the detailed error message ---
                error_msg = (
                    f"\n\n🛑 OUTPUT KEY MISSING IN TRANSFORM {self.__class__.__name__} 🛑\n"
                    f"The transform was expected to produce the key: **'{expected_key}'** "
                    f"(derived from metadata.assign_to mapping) in its returned dictionary.\n"
                    f"Available keys in output: {list(raw_output.keys())}\n\n"
                    f"Potential Solution:\n"
                    f"1. **Check the transform implementation** (`{self.__class__.__name__}.transform_data`) "
                    f"   to ensure it returns a dictionary containing the key '{expected_key}'.\n"
                    f"2. **Check the metadata**: Ensure `assign_to` is structured as a mapping "
                    f"   where the **value** is the key the transform returns."
                )

                # Raise a clearer KeyError, using exception chaining
                raise KeyError(error_msg) from None

        # 3. If the check passes (or is not applicable), return the output
        return raw_output

    return wrapper
