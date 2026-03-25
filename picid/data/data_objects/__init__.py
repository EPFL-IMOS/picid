"""Data containers for transporting data through the pipeline.

Re-exports the main public API for mkdocstrings and direct imports.
"""

from picid.data.data_objects.base import (
    BaseDataObject,
    BaseDataObjectWithMetadata,
    ToDataFrameMixin,
    ToNumpyMixin,
)
from picid.data.data_objects.data import (
    DatasetContainer,
    ExtendedReturnObject,
    NamedDictReturnObject,
    NamedTransformInput,
    ReturnObject,
    SimpleReturnObject,
    SplitUnitCardinality,
    SplitDatasetContainer,
    SplitViewPolicy,
)
from picid.data.data_objects.manifest import ManifestEntry, MetadataManifest
from picid.data.data_objects.slice_info import SliceInfo
from picid.data.data_objects.utils import (
    check_for_nans,
    check_length_consistency,
    convert_to_numpy,
    get_length,
)

__all__ = [
    "BaseDataObject",
    "BaseDataObjectWithMetadata",
    "DatasetContainer",
    "ExtendedReturnObject",
    "ManifestEntry",
    "MetadataManifest",
    "NamedDictReturnObject",
    "NamedTransformInput",
    "ReturnObject",
    "SimpleReturnObject",
    "SliceInfo",
    "SplitUnitCardinality",
    "SplitDatasetContainer",
    "SplitViewPolicy",
    "ToDataFrameMixin",
    "ToNumpyMixin",
    "check_for_nans",
    "check_length_consistency",
    "convert_to_numpy",
    "get_length",
]
