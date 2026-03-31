"""
Transform pipeline: context, steps, checks, and hooks.

This package provides the building blocks used by TransformStrategy (in
strategy/) to run a transform: TransformContext (context.py), all PipelineStep
implementations (steps.py), default before/after checks (checks.py), and the
hook registry / _emit (hooks_registry.py). Import pipeline types and steps
from here or from picid.transforms.base.pipeline.
"""

from picid.transforms.base.pipeline.checks import (
    build_default_after_checks,
    build_default_before_checks,
)
from picid.transforms.base.pipeline.context import (
    AfterCheck,
    AfterCheckFn,
    BeforeCheck,
    BeforeCheckFn,
    PipelineStep,
    TransformContext,
)
from picid.transforms.base.pipeline.hooks_registry import (
    _emit,
    clear_hooks,
    register_hook,
)
from picid.transforms.base.pipeline.unit_metadata import (
    aggregate_unit_metadata,
    drop_unit_metadata,
    preserve_unit_metadata,
)
from picid.transforms.base.pipeline.steps import (
    CopyStep,
    FitStep,
    IntegrityCheckAfterStep,
    IntegrityCheckBeforeStep,
    IntegrityCheckStep,
    MergeStep,
    PostprocessStep,
    RecordManifestStep,
    RegisterChunkBuilderStep,
    TransformStep,
    _get_producer_version,
)

__all__ = [
    "AfterCheck",
    "AfterCheckFn",
    "BeforeCheck",
    "BeforeCheckFn",
    "CopyStep",
    "FitStep",
    "IntegrityCheckAfterStep",
    "IntegrityCheckBeforeStep",
    "IntegrityCheckStep",
    "MergeStep",
    "PipelineStep",
    "PostprocessStep",
    "RecordManifestStep",
    "RegisterChunkBuilderStep",
    "TransformContext",
    "TransformStep",
    "_emit",
    "_get_producer_version",
    "aggregate_unit_metadata",
    "build_default_after_checks",
    "build_default_before_checks",
    "drop_unit_metadata",
    "clear_hooks",
    "preserve_unit_metadata",
    "register_hook",
]
