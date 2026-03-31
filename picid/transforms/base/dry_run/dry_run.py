"""
Structural dry run — validate transform pipeline without executing heavy math.

Checks per transform: apply_to keys available (in initial data or produced by earlier
transforms via assign_to), handler available for (data_kind, capability) when segment
can be built. No tensor computation or fit/transform execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, List, Optional

from picid.data.data_objects import NamedTransformInput, SplitDatasetContainer

from picid.transforms.base.data_kind import get_capability, infer_data_kind
from picid.transforms.base.handlers import get_handler
from picid.transforms.base.transform_manager import ConfigTransformManager


@dataclass
class DryRunResult:
    """Result of dry-run validation for one transform."""

    transform_name: str
    success: bool
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        msg = f"{self.transform_name}: {status}"
        if self.issues:
            msg += " — " + "; ".join(self.issues)
        return msg


def _normalize_keys(keys: Any) -> List[str]:
    """
    Normalize a key specification into a list.

    Parameters
    ----------
    keys : Any
        Key specification that may be ``None``, a string, or an iterable.

    Returns
    -------
    list[str]
        The normalized key list.
    """
    if keys is None:
        return []
    if isinstance(keys, str):
        return [keys]
    return list(keys)


def _get_first_segment(
    container: SplitDatasetContainer,
    apply_to_keys: List[str],
) -> Optional[NamedTransformInput]:
    """
    Build one segment for the first split and first unit.

    Parameters
    ----------
    container : SplitDatasetContainer
        Input container used to extract a representative segment.
    apply_to_keys : list[str]
        Keys used to gather the segment payload.

    Returns
    -------
    NamedTransformInput or None
        A representative segment for data-kind inference, or ``None`` if it
        cannot be constructed.
    """
    if not apply_to_keys:
        return None
    first_key = apply_to_keys[0]
    if first_key not in container:
        return None
    split_data = container[first_key]
    if not isinstance(split_data, Mapping):
        return None
    splits = list(split_data.keys())
    if not splits:
        return None
    split = splits[0]
    try:
        unit_data = {}
        for k in apply_to_keys:
            if (
                k not in container
                or not isinstance(container[k], Mapping)
                or split not in container[k]
            ):
                return None
            val = container[k][split]
            unit_data[k] = val[0] if isinstance(val, list) and len(val) > 0 else val
    except (IndexError, KeyError, TypeError):
        return None
    return NamedTransformInput(**unit_data)


def dry_run_transforms(
    manager: ConfigTransformManager,
    container: SplitDatasetContainer,
) -> List[DryRunResult]:
    """
    Validate that each transform can be applied to the current container.

    Keys produced by earlier transforms are treated as available to later
    transforms. Handler checks are skipped when the apply_to keys are not yet
    present in the initial container, because no representative segment can be
    built for data-kind inference.

    Parameters
    ----------
    manager : ConfigTransformManager
        The transform manager (transform order = pipeline order).
    container : SplitDatasetContainer
        The initial data container (keys present at pipeline start).

    Returns
    -------
    List[DryRunResult]
        One result per transform with its validation status and issues.
    """
    from picid.transforms.base.multisource import find_singular_ragged_dim

    results = []
    transforms = manager.get_data_transforms()
    if not transforms:
        return results

    # Keys available at pipeline start; after each transform we add its assign_to keys
    available_keys = set(container.keys())

    for name, dt in transforms.items():
        issues = []
        apply_to = getattr(dt, "apply_to", None)
        if apply_to is None:
            issues.append("missing apply_to")
            results.append(
                DryRunResult(transform_name=name, success=False, issues=issues)
            )
            continue
        apply_to_list = _normalize_keys(apply_to)
        assign_to_list = _normalize_keys(getattr(dt, "assign_to", None))
        if not assign_to_list and apply_to_list:
            assign_to_list = list(apply_to_list)  # implicit assign_to = apply_to

        # 1) Check apply_to keys are available (in initial container or produced by earlier transforms)
        missing = [k for k in apply_to_list if k not in available_keys]
        if missing:
            issues.append(
                f"apply_to key(s) {missing!r} not available (not in initial data nor produced by earlier transform assign_to)"
            )
            results.append(
                DryRunResult(transform_name=name, success=False, issues=issues)
            )
            continue

        # Keys that are produced by earlier transforms (not in initial container)
        keys_from_earlier = [k for k in apply_to_list if k not in container]
        if keys_from_earlier:
            # Cannot build segment from container for these keys; skip handler check
            issues.append(
                f"apply_to key(s) {keys_from_earlier!r} produced by earlier transform(s); handler check skipped (no data in container)"
            )
            results.append(
                DryRunResult(transform_name=name, success=True, issues=issues)
            )
            available_keys.update(assign_to_list)
            continue

        # 2) All apply_to keys in container — validate structure and handler
        for k in apply_to_list:
            if not isinstance(container.get(k), Mapping):
                issues.append(f"data[{k!r}] is not a split mapping")
        if issues:
            results.append(
                DryRunResult(transform_name=name, success=False, issues=issues)
            )
            available_keys.update(assign_to_list)
            continue

        segment = _get_first_segment(container, apply_to_list)
        if segment is None:
            issues.append("could not build segment for data_kind (missing split/unit?)")
            results.append(
                DryRunResult(transform_name=name, success=False, issues=issues)
            )
            available_keys.update(assign_to_list)
            continue

        try:
            data_kind = infer_data_kind(
                [segment],
                apply_to_list,
                find_singular_ragged_dim,
            )
        except Exception as e:
            issues.append(f"infer_data_kind failed: {e}")
            results.append(
                DryRunResult(transform_name=name, success=False, issues=issues)
            )
            available_keys.update(assign_to_list)
            continue

        try:
            capability = get_capability(dt.transform_instance)
        except Exception as e:
            issues.append(f"get_capability failed: {e}")
            results.append(
                DryRunResult(transform_name=name, success=False, issues=issues)
            )
            available_keys.update(assign_to_list)
            continue

        try:
            get_handler(data_kind, capability)
        except KeyError:
            issues.append(
                f"no handler for (data_kind={data_kind!r}, capability={capability!r})"
            )
            results.append(
                DryRunResult(transform_name=name, success=False, issues=issues)
            )
            available_keys.update(assign_to_list)
            continue

        results.append(DryRunResult(transform_name=name, success=True, issues=[]))
        available_keys.update(assign_to_list)

    return results
