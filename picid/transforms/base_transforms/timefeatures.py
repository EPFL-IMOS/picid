from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Assuming this import path is correct for your project
from picid.transforms.base.base_transform import DenseTransform
from picid.data.data_objects.data import NamedTransformInput


class TimePFNTimeFeaturesTransform(DenseTransform):
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

    def __init__(self, freq: str, **kwargs):
        # Call the BaseTransform constructor.
        # Your BaseTransform doesn't take specific 'apply_to'/'fit_on' in its init,
        # so just pass any generic kwargs along.
        super().__init__()

        # Store the frequency, which is specific to this transform
        self.freq = freq

        # Store initial kwargs for __repr__, if BaseTransform doesn't already do this comprehensively
        # (your provided BaseTransform doesn't show _init_kwargs in its __init__,
        # but your original TimePFNTimeFeaturesTransform did, so keeping it consistent for your repr)
        self._init_kwargs = kwargs

    def __repr__(self):
        # Ensure 'freq' is included in the representation, along with any other init args
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}(freq={self.freq!r}, {args})"

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Any:
        r"""Applies the time features extraction to the input data.

        This method takes the input data segment (expected to contain timestamps),
        converts them to datetime objects if necessary, and then uses the
        `time_features` utility to generate new features.

        Parameters
        ----------
        data : NamedTransformInput
            The input data, expected to be a sequence of timestamps or a structure
            from which timestamps can be extracted (e.g., a DataFrame).

        Returns
        -------
        np.ndarray
            A NumPy array containing the extracted time features.
        """
        data_segment = data["timestamps"]
        # Your original working logic: Convert input to DataFrame and extract 'date' values.
        # This assumes the 'data_segment' can be directly converted to a DataFrame
        # and has a 'date' column, or can be implicitly handled by pd.to_datetime.
        initial_dates = pd.DataFrame(pd.to_datetime(data_segment))

        # Call your existing time_features function with the extracted dates and frequency.
        # This function is expected to return a structure that can be converted to a DataFrame
        # and then transposed.
        try:
            df_stamp_raw = time_features(
                pd.to_datetime(initial_dates["date"].values), freq=self.freq
            )
        except:
            df_stamp_raw = time_features(
                pd.to_datetime(initial_dates["datetime"].values), freq=self.freq
            )

        # Convert the result to a DataFrame and transpose, as per your original working code.
        transformed_data = pd.DataFrame(df_stamp_raw.transpose(1, 0))
        return transformed_data

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data: Any) -> Any:
        return self.transform_data(data)


# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq="h"):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
