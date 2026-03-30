"""Conversion mixins shared by data-object and container classes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from picid.data.data_objects.utils import convert_to_numpy


class ToNumpyMixin:
    """Provide on-demand conversion of stored values to NumPy arrays."""

    def to_numpy(self, key: str, ensure_2d: bool = True) -> np.ndarray:
        """
        Convert the data for a given key to a NumPy array on demand.

        Parameters
        ----------
        key : str
            Name of the stored item to convert.
        ensure_2d : bool, default=True
            Whether to require the returned array to be two-dimensional.

        Returns
        -------
        numpy.ndarray
            Converted NumPy representation of the requested item.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in OrderedInput.")
        value = self[key]
        return convert_to_numpy(value, ensure_2d)


class ToDataFrameMixin:
    """Provide on-demand conversion of stored values to pandas data frames."""

    def to_dataframe(self, key: str) -> pd.DataFrame:
        """
        Return a pandas.DataFrame representation of the data for a given key.

        Parameters
        ----------
        key : str
            Name of the stored item to convert.

        Returns
        -------
        pandas.DataFrame
            Data-frame representation of the requested item.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in DataChunk.")

        value = self[key]
        if isinstance(value, pd.DataFrame):
            return value

        array_data = self.to_numpy(key, ensure_2d=False)
        columns = getattr(self, "metadata", {}).get("column_map", {}).get(key)
        return pd.DataFrame(array_data, columns=columns)
