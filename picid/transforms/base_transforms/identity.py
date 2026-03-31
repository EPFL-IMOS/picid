from typing import Any
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin


class IdentityPassThrough(NoFitPerSegmentMixin, DenseTransform):
    """
    Pass input data through unchanged.

    Parameters
    ----------
    *args : tuple
        Ignored positional compatibility arguments.
    **kwargs : dict
        Ignored keyword compatibility arguments.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the identity transform.

        Parameters
        ----------
        *args : tuple
            Ignored positional compatibility arguments.
        **kwargs : dict
            Ignored keyword compatibility arguments.
        """
        super().__init__()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """
        Do nothing because the transform has no learned state.

        Parameters
        ----------
        data : NamedTransformInput
            Input payload dictionary.
        metadata : dict
            Transform metadata.
        """
        pass

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
        """
        Return the input data object unchanged.

        Parameters
        ----------
        data : NamedTransformInput
            Input payload dictionary.
        metadata : dict
            Transform metadata.

        Returns
        -------
        Any
            The input payload unchanged.
        """
        return data
