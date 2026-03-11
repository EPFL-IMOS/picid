"""
Transform handler protocol and registry (data_kind × capability).

Handlers implement fit_prepare and transform_apply for a (DataKind, TransformCapability)
pair. The multisource mixins call get_handler(data_kind, capability) and then
run the handler; this avoids branching inside mixins and keeps ragged/dense
logic in one place. See handlers.py for DenseDenseHandler, RaggedDenseHandler,
RaggedRaggedHandler.
"""

from picid.transforms.base.handlers.handlers import (
    DenseDenseHandler,
    RaggedDenseHandler,
    RaggedRaggedHandler,
    TransformHandler,
    get_handler,
)

__all__ = [
    "DenseDenseHandler",
    "RaggedDenseHandler",
    "RaggedRaggedHandler",
    "TransformHandler",
    "get_handler",
]
