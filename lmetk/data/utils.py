import pandas as pd
import numpy as np
import torch
from typing import Optional, Union


def time_features(
    dates: pd.Series, use_features: Optional[list[str]] = None, normalize: bool = True, return_float=True,
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

    return df[use_features], dates


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


def to_torch_tensor(data, device=None):
    """
    Convert input data (list, numpy array, or tensor) to a torch.Tensor
    with float32 if floating type, long if integer type.

    Args:
        data: list, numpy.ndarray, or torch.Tensor
        device (torch.device or str, optional): target device ('cpu', 'cuda', etc.)

    Returns:
        torch.Tensor
    """
    if torch.is_tensor(data):
        # If already a tensor, just cast
        if data.dtype.is_floating_point:
            return data.to(dtype=torch.float32, device=device)
        elif data.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return data.to(dtype=torch.long, device=device)
        else:
            raise TypeError(f"Unsupported tensor dtype: {data.dtype}")

    # Convert to NumPy array without copying if possible
    arr = np.array(data, copy=False)

    if np.issubdtype(arr.dtype, np.floating):
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
    elif np.issubdtype(arr.dtype, np.integer):
        return torch.as_tensor(arr, dtype=torch.long, device=device)
    else:
        raise TypeError(f"Unsupported input dtype: {arr.dtype}")