from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from lmetk.data.data_objects.data import OrderedInput
from lmetk.transforms.base.base_transform import BaseTransform

# Assuming this import path is correct for your project


class RailwayTimeFeaturesTransform(BaseTransform):
    r"""Extracts time features from the input data.

    This transform processes time-series data to generate additional features
    based on the timestamp, using the `time_features` utility.

    Parameters
    ----------
    freq : str
        The frequency string (e.g., 'h' for hourly, 'd' for daily)
        to be used by the `time_features` function.
    **kwargs : optional
        Additional parameters for the base transform.
    """

    def __init__(
        self, use_features: List[str], normalize_time_features: bool, **kwargs
    ):
        # Call the BaseTransform constructor.
        # Your BaseTransform doesn't take specific 'apply_to'/'fit_on' in its init,
        # so just pass any generic kwargs along.
        super().__init__(**kwargs)

        # Store the frequency, which is specific to this transform
        self.use_features = use_features
        self.normalize_time_features = normalize_time_features
        # Store initial kwargs for __repr__, if BaseTransform doesn't already do this comprehensively
        # (your provided BaseTransform doesn't show _init_kwargs in its __init__,
        # but your original TimePFNTimeFeaturesTransform did, so keeping it consistent for your repr)
        self._init_kwargs = kwargs

    def __repr__(self):
        # Ensure 'freq' is included in the representation, along with any other init args
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}(freq={self.freq!r}, {args})"

    def transform_data(self, data: OrderedInput, metadata: Dict) -> Any:
        r"""Applies the time features extraction to the input data.

        This method takes the input data segment (expected to contain timestamps),
        converts them to datetime objects if necessary, and then uses the
        `time_features` utility to generate new features.

        Parameters
        ----------
        data_segment : Any
            The input data, expected to be a sequence of timestamps or a structure
            from which timestamps can be extracted (e.g., a DataFrame).

        Returns
        -------
        np.ndarray
            A NumPy array containing the extracted time features.
        """
        # Your original working logic: Convert input to DataFrame and extract 'date' values.
        # This assumes the 'data_segment' can be directly converted to a DataFrame
        # and has a 'date' column, or can be implicitly handled by pd.to_datetime.
        data_segment = data

        # make sure timestamps are in data_segment
        assert (
            "timestamps" in data_segment.keys()
        ), "timestamps key is not in data_segment {data_segment}, add timestamps into apply_to argument"
        df_stamp_raw = time_features(
            data_segment["timestamps"],
            use_features=self.use_features,
            normalize=self.normalize_time_features,
        )

        # Check is there are any nan values
        if df_stamp_raw.isnull().values.any():
            raise ValueError("NaN values found in df_stamp_raw")

        # Return the values as a NumPy array.
        return df_stamp_raw.values

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data: Any, metadata: Dict) -> Any:
        return self.transform_data(data, metadata=metadata)

    def transform(self, data: Any, metadata: Dict) -> Any:
        return self(data, metadata=metadata)


def time_features(
    dates: pd.Series,
    use_features: Optional[list[str]] = None,
    normalize: bool = True,
    return_float=True,
) -> tuple[pd.DataFrame, pd.Series]:
    if use_features is None:
        use_features = ["year", "month", "day", "weekday", "hour", "minute"]

    if not isinstance(dates, pd.Series):
        raise ValueError("dates should be a pandas Series")

    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)

    df = pd.DataFrame()

    if "year" in use_features:
        years = dates.dt.year
        if normalize:
            max_year, min_year = years.max(), years.min()
            df["year"] = (years - min_year) / max(1.0, (max_year - min_year))
        else:
            df["year"] = years

    if "month" in use_features:
        months = dates.dt.month
        df["month"] = (months - 1) / 11.0 if normalize else months

    if "day" in use_features:
        days = dates.dt.day
        if normalize:
            df["day"] = (days - 1) / (dates.dt.days_in_month - 1)
        else:
            df["day"] = days

    if "weekday" in use_features:
        weekdays = dates.dt.weekday
        df["weekday"] = weekdays / 6.0 if normalize else weekdays

    if "hour" in use_features:
        hours = dates.dt.hour
        df["hour"] = hours / 23.0 if normalize else hours

    if "minute" in use_features:
        minutes = dates.dt.minute
        df["minute"] = minutes / 59.0 if normalize else minutes

    # Check for NaNs and zero-columns (all values are zero)
    for col in df.columns:
        if df[col].isnull().any():
            raise ValueError(f"Column {col} contains NaN values.")
        if (df[col] == 0).all():
            raise ValueError(f"Column {col} is a zero column.")
    # make all the time features to be float
    if return_float:
        df = df.astype(float)

    return df[use_features]


def reconstruct_datetime_from_features(
    data: Union[np.ndarray, pd.DataFrame], columns: list[str]
) -> pd.Series:
    """
    Reconstruct datetime from features (either DataFrame or NumPy array).

    Parameters:
    - data: pd.DataFrame or np.ndarray containing time components
    - columns: List of column names corresponding to the input

    Returns:
    - pd.Series of datetime64[ns]
    """

    # Convert float columns to int if necessary
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.floating):
            data = np.round(data).astype(int)
    elif isinstance(data, pd.DataFrame):
        float_cols = data.select_dtypes(include=["float", "float64", "float32"]).columns
        if len(float_cols) > 0:
            data[float_cols] = data[float_cols].round().astype(int)

    # Convert numpy array to DataFrame using provided column names
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("NumPy array must be 2D")
        if data.shape[1] != len(columns):
            raise ValueError("Number of columns in array and `columns` must match")
        df = pd.DataFrame(data, columns=columns)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        df = df[columns]
    else:
        raise TypeError("Input must be a numpy.ndarray or pandas.DataFrame")

    # Fill missing optional fields
    df_full = {
        "year": df["year"] if "year" in df else 1970,
        "month": df["month"] if "month" in df else 1,
        "day": df["day"] if "day" in df else 1,
        "hour": df["hour"] if "hour" in df else 0,
        "minute": df["minute"] if "minute" in df else 0,
        "second": df["second"] if "second" in df else 0,
    }

    return pd.to_datetime(pd.DataFrame(df_full))
