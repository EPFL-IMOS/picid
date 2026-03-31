"""
Base transforms, pipeline, and data transforms.

Re-export the main public API for mkdocstrings and direct imports.
"""

from picid.transforms.base.base_transform import (
    BaseTransform,
    DenseTransform,
    RaggedOrDenseTransform,
    RaggedTransform,
)
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base.pipeline import (
    TransformContext,
    PipelineStep,
)
from picid.transforms.base.strategy import (
    TransformStrategy,
    postprocess_transformed_data,
)
from picid.transforms.base.transform_manager import (
    ConfigTransformManager,
    create_transform_manager_from_config,
)
from picid.transforms.base.transform_pipeline import (
    TransformPipeline,
    TransformSequenceProtocol,
)

__all__ = [
    "BaseTransform",
    "ConfigTransformManager",
    "DataTransform",
    "DenseTransform",
    "PipelineStep",
    "RaggedOrDenseTransform",
    "RaggedTransform",
    "TransformContext",
    "TransformPipeline",
    "TransformSequenceProtocol",
    "TransformStrategy",
    "create_transform_manager_from_config",
    "postprocess_transformed_data",
]
