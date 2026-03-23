"""Abstract preprocessing contract.

This module defines the interface that concrete preprocessors must implement
when orchestrating datasource loading, transform application, and manifest
generation. The preprocessing layer depends on the datasource protocol rather
than concrete loader classes so the data layer can evolve independently.
"""

from abc import ABC, abstractmethod

from numpy import ndarray

from picid.data.datasources.base.contracts import DatasourceProtocol
from picid.transforms.base.transform_pipeline import TransformSequenceProtocol


class PreProcessorInterface(ABC):
    """Abstract preprocessing entry point.

    Parameters
    ----------
    datasource : DatasourceProtocol
        Datasource implementation used to load and split the raw dataset.
    transforms : TransformSequenceProtocol
        Ordered transform sequence applied after the datasource has been
        materialized.
    **kwargs : dict[str, object]
        Reserved for concrete preprocessors that need additional options.
    """

    def __init__(
        self,
        datasource: DatasourceProtocol,
        transforms: TransformSequenceProtocol,
        **kwargs,
    ):
        """Store the datasource and transform sequence for later execution."""
        self.datasource = datasource
        self.transforms = transforms

    @abstractmethod
    def get_processed_data_dict(
        self,
        return_splits_on_first_level: bool = False,
    ) -> dict[str, dict[str, ndarray | list[ndarray]]]:
        """Return the processed dataset as a nested split dictionary.

        Parameters
        ----------
        return_splits_on_first_level : bool, optional
            When True, the split name becomes the first dictionary key. When
            False, the returned structure is keyed by data tensor name first.

        Returns
        -------
        dict[str, dict[str, numpy.ndarray | list[numpy.ndarray]]]
            Final processed data grouped by tensor name and split.
        """
        raise NotImplementedError("get_processed_data method not implemented.")

    @abstractmethod
    def get_meta_data_dict(self):
        """Return metadata collected during preprocessing."""
        raise NotImplementedError("get_meta_data_dict method not implemented.")

    @abstractmethod
    def fetch_data(self) -> None:
        """Materialize the datasource payload into preprocessor state."""
        raise NotImplementedError("prepare_data method not implemented.")

    @abstractmethod
    def apply_transforms(self) -> None:
        """Apply the configured transform sequence to the loaded data."""
        raise NotImplementedError("apply_transforms method not implemented.")

    @abstractmethod
    def pipeline(self):
        """Execute the full preprocessing pipeline."""
        raise NotImplementedError("pipeline method not implemented.")
