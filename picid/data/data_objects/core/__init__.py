"""Core data-object primitives shared across container types."""

from picid.data.data_objects.core.base_data_object import BaseDataObject
from picid.data.data_objects.core.metadata_data_object import BaseDataObjectWithMetadata

__all__ = [
    "BaseDataObject",
    "BaseDataObjectWithMetadata",
]
