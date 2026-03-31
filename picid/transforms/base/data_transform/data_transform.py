"""
data_transform.py
=================

DataTransform: facade that configures a transform and delegates execution
to TransformStrategy.

Changes from the original
-------------------------
* _validate_forward_output is fully removed. Structural validation (type,
  keys, splits) is performed by IntegrityCheckAfterStep's default after-checks
  when validate_output=True. Soft diagnostic warnings (type change, unit-count
  change) live in MergeStep where the before/after data is locally available.

* forward() delegates entirely to strategy.apply(); no post-pipeline logic here.

* All other logic — apply_to / assign_to processing, fit_on_split resolution,
  transform_data signature validation, Hydra instantiation — is unchanged.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, get_origin

import omegaconf
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from picid.data.data_objects import NamedTransformInput, SplitDatasetContainer
from picid.transforms.base.base_transform import BaseTransform
from picid.transforms.base.strategy import TransformStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------


def is_sequence(obj):
    """
    Return whether an object should be treated as a non-string sequence.

    Parameters
    ----------
    obj : Any
        Object to inspect.

    Returns
    -------
    bool
        ``True`` for list/tuple/array-like objects, but ``False`` for
        ``str`` and ``bytes``.
    """
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def _format_signature(param_set):
    return "(" + ", ".join(f"{n}: {t.__name__}" for n, t in param_set) + ")"


def _type_matches(ann, expected):
    if ann is inspect._empty:
        return False
    if ann is expected or ann == expected:
        return True
    origin = get_origin(ann)
    if origin is not None:
        return origin is expected
    try:
        return issubclass(ann, expected)
    except TypeError:
        return False


def _matches(sig_params, expected_params):
    if len(sig_params) != len(expected_params):
        return False
    for (name, param), (exp_name, exp_type) in zip(sig_params, expected_params):
        if name != exp_name:
            return False
        if not _type_matches(param.annotation, exp_type):
            return False
    return True


# ---------------------------------------------------------------------------
# DataTransform
# ---------------------------------------------------------------------------


class DataTransform:
    """
    Facade that configures a transform and delegates to TransformStrategy.

    Structural validation is owned entirely by the pipeline
    (IntegrityCheckAfterStep) when ``validate_output=True``. This facade keeps
    configuration parsing and runtime delegation in one place.

    Parameters
    ----------
    transform_name : str
        Unique name for this transformation step.
    transform : BaseTransform or DictConfig
        Transform instance or Hydra config used to instantiate one.
    metadata : dict
        Configuration dictionary controlling apply/assign keys, fitting,
        validation, and optional custom integrity checks.

    Attributes
    ----------
    transform_name : str
        Unique name for this transformation step.
    transform_instance : BaseTransform
        Instantiated transform object.
    metadata : dict
        Configuration metadata for the transform.
    apply_to : list[str]
        Keys read from the container.
    apply_to_map : list[str] or None
        Optional mapping-style view of ``apply_to``.
    assign_to : list[str]
        Keys written by the transform.
    assign_to_map : list[str]
        Mapping-style output key list used by the postprocess step.
    fit_on_split : str or None
        Split used for fitting when applicable.
    fit_on_key : str or None
        Key used for fitting when applicable.
    transform_on_keys : list[str] or None
        Splits processed during transformation when restricted.
    multi_source_fit : bool
        Whether the transform should use multi-source fitting.
    validate_output : bool
        Whether the default integrity checks run.
    include_slice_info_in_metadata : bool
        Whether slice information is injected into metadata.
    integrity_checks_before : list[tuple] or None
        Custom before-checks, if provided.
    integrity_checks_after : list[tuple] or None
        Custom after-checks, if provided.
    strategy : TransformStrategy
        Strategy responsible for executing the transform.
    """

    # ------------------------------------------------------------------
    # apply_to / assign_to processing (unchanged)
    # ------------------------------------------------------------------

    def _process_apply_to(self, apply_to):
        if apply_to is None:
            raise ValueError(
                f"Metadata for transform '{self.transform_name}' must specify 'apply_to'."
            )
        if isinstance(apply_to, omegaconf.listconfig.ListConfig):
            apply_to = list(apply_to)

        apply_to_map = None
        if isinstance(apply_to, Mapping):
            apply_to_map = list(apply_to.keys())
            apply_to = list(apply_to.values())

        return apply_to, apply_to_map

    def _process_assign_to(self, assign_to, apply_to):
        if assign_to is None:
            if is_sequence(apply_to):
                logger.info(
                    "Implicit assign_to for transform '%s': %s",
                    self.transform_name,
                    apply_to,
                )
            elif isinstance(apply_to, Mapping):
                raise ValueError(
                    f"Transform '{self.transform_name}': apply_to is a mapping "
                    "but assign_to is not specified. Specify assign_to explicitly."
                )
            assign_to = apply_to
            assign_to_map = apply_to

        elif isinstance(assign_to, Mapping):
            assign_to_map = list(assign_to.keys())
            assign_to = list(assign_to.values())

        elif isinstance(assign_to, str):
            assign_to = [assign_to]
            assign_to_map = assign_to

        elif is_sequence(assign_to):
            assign_to_map = list(assign_to)

        else:
            raise ValueError(
                f"Transform '{self.transform_name}': unsupported assign_to "
                f"type {type(assign_to)}."
            )

        return assign_to, assign_to_map

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, transform_name: str, transform, metadata: dict):
        """
        Parameters
        ----------
        transform_name : str
            Unique name for this transformation step.
        transform : BaseTransform or DictConfig
            The transform instance (or Hydra config to instantiate one).
        metadata : dict
            Configuration dict.  Recognised keys:

            apply_to : str | list | dict  (required)
                Key(s) in the data container to read from.
            assign_to : str | list | dict  (optional)
                Key(s) to write results to; defaults to apply_to.
            fit_on : str  (optional)
                Split to fit on ("train" | "val" | "test").
            fit_on_key : str  (optional)
                Key to fit on; defaults to apply_to when fit_on is set.
            transform_on_keys : list  (optional)
                Restrict transformation to these splits.
            validate_output : bool  (default True)
                When True, default structural before/after checks run.
            include_slice_info_in_metadata : bool  (optional, default False)
                When True, the pipeline adds the container's slice (split, unit_ids,
                cycle_ids, bounds, index_map) to the metadata dict passed to
                transform_data and fit_data, under the key "slice_info". Use when
                the transform needs to know which slice of the dataset it is processing.
            integrity_checks_before : list  (optional)
                Custom before-checks (name, fn, policy) triples.
            integrity_checks_after : list  (optional)
                Custom after-checks (name, fn, policy) triples.
        """
        self.transform_name = transform_name
        self.metadata = metadata

        self.apply_to = metadata.get("apply_to", None)
        self.apply_to, self.apply_to_map = self._process_apply_to(self.apply_to)

        assign_to = metadata.get("assign_to", None)
        self.assign_to, self.assign_to_map = self._process_assign_to(
            assign_to, self.apply_to
        )

        self.fit_on_split = metadata.get("fit_on", None)
        assert self.fit_on_split in (None, "train", "val", "test"), (
            f"fit_on must be None, 'train', 'val', or 'test'; "
            f"got {self.fit_on_split!r}."
        )

        self.fit_on_key = metadata.get("fit_on_key", None)
        self.transform_on_keys = metadata.get("transform_on_keys", None)

        if self.fit_on_split is not None and self.fit_on_key is None:
            if isinstance(self.apply_to, list) and len(self.apply_to) > 1:
                raise ValueError(
                    f"Transform '{transform_name}': apply_to has multiple keys "
                    "but fit_on_key is not specified. Set fit_on_key explicitly."
                )
            # Single apply_to key: use it as fit_on_key.
            self.fit_on_key = (
                self.apply_to[0] if isinstance(self.apply_to, list) else self.apply_to
            )

        self.multi_source_fit = metadata.get("multi_source_fit", False)

        # validate_output=True: default structural before/after checks run.
        # validate_output=False: no defaults; custom lists still apply if provided.
        self.validate_output: bool = metadata.get("validate_output", True)
        self.include_slice_info_in_metadata: bool = metadata.get(
            "include_slice_info_in_metadata", False
        )

        # Custom integrity checks (set programmatically or from config).
        self.integrity_checks_before: Optional[List[Tuple]] = metadata.get(
            "integrity_checks_before", None
        )
        self.integrity_checks_after: Optional[List[Tuple]] = metadata.get(
            "integrity_checks_after", None
        )

        # Ensure we have a BaseTransform.
        if not isinstance(transform, (BaseTransform, DictConfig)):
            raise TypeError(
                f"Transform '{transform_name}' (type: {type(transform)}) "
                "must be a BaseTransform instance or a Hydra DictConfig."
            )

        self.strategy = TransformStrategy()

        if isinstance(transform, DictConfig):
            self.transform_instance: BaseTransform = instantiate(transform)
        else:
            self.transform_instance = transform

        if not isinstance(self.transform_instance, BaseTransform):
            raise TypeError(
                f"Transform '{transform_name}' (type: {type(self.transform_instance)}) "
                "must inherit from BaseTransform."
            )

        # --- transform_data signature validation (unchanged) ---
        is_kwargs_apply_to = self.apply_to_map is not None
        allowed_param_sets = [
            [("data", NamedTransformInput), ("metadata", dict)],
            [("data", Any), ("metadata", dict)],
        ]
        sig = inspect.signature(self.transform_instance.transform_data)
        sig_params = list(sig.parameters.items())

        if not any(_matches(sig_params, params) for params in allowed_param_sets):
            if is_kwargs_apply_to:
                named_params = [n for n, _ in sig_params]
                assert "metadata" in named_params, (
                    f"Transform '{transform_name}': transform_data must include "
                    "a 'metadata' parameter."
                )
                assert len(self.apply_to_map) == len(named_params) - 1, (
                    f"Transform '{transform_name}': transform_data must have "
                    f"{len(self.apply_to_map)} data parameter(s) for "
                    f"apply_to_map={self.apply_to_map}, got {len(named_params) - 1}."
                )
                assert all(k in named_params for k in self.apply_to_map), (
                    f"Transform '{transform_name}': transform_data parameters "
                    f"must include {self.apply_to_map}."
                )
            else:
                allowed_str = " or ".join(
                    _format_signature(p) for p in allowed_param_sets
                )
                got_str = (
                    "("
                    + ", ".join(
                        f"{n}: {getattr(p.annotation, '__name__', str(p.annotation))}"
                        for n, p in sig_params
                    )
                    + ")"
                )
                raise TypeError(
                    f"Transform '{transform_name}': transform_data signature "
                    f"must be {allowed_str}, got {got_str}."
                )

    # ------------------------------------------------------------------
    # forward — full delegation to strategy; no post-pipeline logic
    # ------------------------------------------------------------------

    def forward(
        self,
        data: SplitDatasetContainer,
    ) -> Tuple[SplitDatasetContainer, Dict[str, Any]]:
        """
        Apply the transform to data via TransformStrategy.

        Structural validation (type, keys, splits) runs inside the pipeline
        as IntegrityCheckAfterStep when validate_output=True. No validation
        logic lives in this method.

        Parameters
        ----------
        data : SplitDatasetContainer
            Input data with structure: key → split → [unit_array, …].

        Returns
        -------
        transformed_data : SplitDatasetContainer
            Data container with transform results merged in.
        transform_log : Dict[str, Any]
            Per-split logs from transform_multi_source.
        """
        if self.apply_to_map is not None:
            raise NotImplementedError(
                "apply_to as a mapping (kwargs-style transform_data) is not yet "
                "supported in DataTransform.forward."
            )

        return self.strategy.apply(
            transform_instance=self.transform_instance,
            data=data,
            apply_to_keys=self.apply_to,
            assign_to_keys=self.assign_to,
            assign_to_keys_map=self.assign_to_map,
            fit_on_split=self.fit_on_split,
            fit_on_key=self.fit_on_key,
            transform_on_keys=self.transform_on_keys,
            step_id=self.transform_name,
            integrity_checks_before=self.integrity_checks_before,
            integrity_checks_after=self.integrity_checks_after,
            validate_output=self.validate_output,
            include_slice_info_in_metadata=self.include_slice_info_in_metadata,
        )
