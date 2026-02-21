from typing import Any
from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin


class IdentityPassThrough(NoFitPerSegmentMixin, DenseTransform):
    """
    An identity transform that passes the input data straight through
    without modification in both the fit and transform stages.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the transform. Accepts any arguments but ignores them."""
        super().__init__()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """
        Does nothing, as no state needs to be learned for an identity operation.
        """
        pass

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
        """
        Returns the input data object without any transformation.
        """
        return data
