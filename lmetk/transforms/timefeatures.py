"""Identity transform that does nothing to the input data."""
import pandas as pd
from lmetk.data.timefeatures import time_features

class TimePFNTimeFeaturesTransform():
    r"""An identity transform that does nothing to the input data.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        
        # Auto-assign keyword arguments to instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}({args})"

    def __call__(self, data: dict):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        dict
            The same data.
        """
        df_stamp = data['timestamps']

        initial_dates = pd.DataFrame(pd.to_datetime(df_stamp))
        df_stamp = time_features(pd.to_datetime(initial_dates['date'].values), freq=self.freq)
        df_stamp = pd.DataFrame(df_stamp.transpose(1, 0))

        data['timestamps'] = df_stamp
        data['initial_dates'] = initial_dates

        return data