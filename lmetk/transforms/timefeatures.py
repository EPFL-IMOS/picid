import pandas as pd
import numpy as np
from typing import Any, List

# Assuming this import path is correct for your project
from lmetk.data.timefeatures import time_features 
from lmetk.transforms.base_transform import BaseTransform 

class TimePFNTimeFeaturesTransform(BaseTransform):
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

    def fit_single_source(self, data_segment: Any):
        """
        Time feature extraction is a stateless operation, so no fitting is required.
        This method does nothing.
        """
        pass

    def fit_multi_source(self, list_of_data_segments: List[Any]):
        """
        Time feature extraction is a stateless operation, so no fitting is required
        across multiple sources. This method does nothing.
        """
        pass

    def transform_data(self, data_segment: Any, mode: str) -> Any:
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
        initial_dates = pd.DataFrame(pd.to_datetime(data_segment))
        
        # Call your existing time_features function with the extracted dates and frequency.
        # This function is expected to return a structure that can be converted to a DataFrame
        # and then transposed.
        try:
            df_stamp_raw = time_features(pd.to_datetime(initial_dates['date'].values), freq=self.freq)
        except:
            df_stamp_raw = time_features(pd.to_datetime(initial_dates['datetime'].values), freq=self.freq)
        
        # Convert the result to a DataFrame and transpose, as per your original working code.
        df_stamp_processed = pd.DataFrame(df_stamp_raw.transpose(1, 0))
        
        # Return the values as a NumPy array.
        return df_stamp_processed.values

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data: Any) -> Any:
        return self.transform_data(data)
    
    def transform(self, data: Any, mode: str) -> Any:
        return self(data)

# """Identity transform that does nothing to the input data."""
# import pandas as pd
# from lmetk.data.timefeatures import time_features

# class TimePFNTimeFeaturesTransform():
#     r"""An identity transform that does nothing to the input data.

#     Parameters
#     ----------
#     **kwargs : optional
#         Parameters for the base transform.
#     """

#     def __init__(self, **kwargs):
#         super().__init__()
        
#         # Auto-assign keyword arguments to instance attributes
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#     def __repr__(self):
#         args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
#         return f"{self.__class__.__name__}({args})"

#     def __call__(self, data):
#         r"""Apply the transform to the input data.

#         Parameters
#         ----------
#         data : 
#             The input data.

#         Returns
#         -------
#         dict
#             The same data.
#         """

#         initial_dates = pd.DataFrame(pd.to_datetime(data))
#         df_stamp = time_features(pd.to_datetime(initial_dates['date'].values), freq=self.freq)
#         df_stamp = pd.DataFrame(df_stamp.transpose(1, 0))
#         return df_stamp.values
    
#     def transform(self, data):
#         return self(data)