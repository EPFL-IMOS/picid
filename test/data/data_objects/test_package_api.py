"""Public import-surface checks for the split ``data_objects`` package."""

from picid.data.data_objects import (
    BaseDataObject,
    BaseDataObjectWithMetadata,
    DatasetContainer,
    NamedTransformInput,
    SplitDatasetContainer,
    SplitUnitCardinality,
    SplitViewPolicy,
)
from picid.data.data_objects.containers import (
    DatasetContainer as DatasetContainerFromContainers,
)
from picid.data.data_objects.containers import (
    SplitDatasetContainer as SplitDatasetContainerFromContainers,
)
from picid.data.data_objects.core import (
    BaseDataObject as BaseDataObjectFromCore,
)
from picid.data.data_objects.core import (
    BaseDataObjectWithMetadata as BaseDataObjectWithMetadataFromCore,
)
from picid.data.data_objects.returns import (
    NamedTransformInput as NamedTransformInputFromReturns,
)
from picid.data.data_objects.types import (
    SplitUnitCardinality as SplitUnitCardinalityFromTypes,
)
from picid.data.data_objects.types import (
    SplitViewPolicy as SplitViewPolicyFromTypes,
)
from picid.data.data_objects.validation import (
    build_split_alignment_report_table,
    collect_split_alignment_report,
    describe_unit_payload,
)


def test_package_root_re_exports_split_data_objects():
    """The package root should re-export the main public container surface."""
    assert BaseDataObject is BaseDataObjectFromCore
    assert BaseDataObjectWithMetadata is BaseDataObjectWithMetadataFromCore
    assert DatasetContainer is DatasetContainerFromContainers
    assert SplitDatasetContainer is SplitDatasetContainerFromContainers
    assert NamedTransformInput is NamedTransformInputFromReturns
    assert SplitUnitCardinality is SplitUnitCardinalityFromTypes
    assert SplitViewPolicy is SplitViewPolicyFromTypes


def test_validation_subpackage_exports_split_report_helpers():
    """The validation package should expose the split-report helper functions."""
    assert callable(build_split_alignment_report_table)
    assert callable(collect_split_alignment_report)
    assert callable(describe_unit_payload)
