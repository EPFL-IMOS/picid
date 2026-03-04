from copy import deepcopy
from typing import Literal, Callable

import numpy as np
from pandas import DataFrame

from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.datasources.base.single_source_loader import SingleSourceLoader


class CustomSingleSourceLoader(SingleSourceLoader):
    """
    A data source loader class designed to handle single-source data from either CSV (DataFrame)
    or NumPy formats, supporting custom splitting and task-specific formatting.
    """

    def __init__(self,
                 data,
                 target_column: int | str,
                 task_mode: str,
                 loading_type: Literal['csv', 'numpy'],
                 is_part_of_multisource: bool = False,
                 data_splitter: Callable[[dict], dict] = None,
                 data_name: str = 'custom single source datasource',
                 **kwargs):
        """
                Initializes the CustomDataSource with the raw data and configuration parameters.

                Args:
                    data: The raw input data (typically a pandas DataFrame or numpy ndarray).
                    target_column (int | str): The index or name of the column containing the target feature.
                    task_mode (str): The mode of the task (e.g., 'classification', 'regression').
                    loading_type (Literal['csv', 'numpy']): Specifies the internal loading logic to use.
                    is_part_of_multisource (bool): Flag indicating if this source is part of a larger collection.
                    data_splitter (Callable[[dict], dict], optional): A function used to partition data into subsets.
                        It must take as input a dictionary containing all features and targets,
                        and must return a mask for each split (train, test, and dev)
                    data_name (str): A descriptive name for the data source.
                    **kwargs: Additional arguments passed to the SingleSourceLoader parent class.
        """

        super().__init__(
            is_part_of_multisource=is_part_of_multisource,
            data_splitter=data_splitter,
            data_name=data_name,
            task_mode=task_mode,
            **kwargs)

        if loading_type == 'csv':
            self._loading_function = self._load_csv
        elif loading_type == 'numpy':
            if target_column < 0:
                target_column = np.arange(data.shape[1])[target_column]
            self._loading_function = self._load_numpy
        else:
            assert False, 'loading_type must be either "csv" or "numpy".'

        self._inner_data = data
        self._target_column = target_column
        self._task_mode = task_mode

    def _load_csv(self):
        """
                Internal method to extract features and targets from a pandas DataFrame.

                Returns:
                    tuple: A tuple containing (features_dataframe, target_series).
        """

        df_target = self._inner_data[self._target_column].to_frame().copy()
        data = self._inner_data.drop(columns=[self._target_column], inplace=False)

        return data, df_target

    def _load_numpy(self):
        """
                Internal method to extract features and targets from a NumPy array.

                Returns:
                    tuple: A tuple containing (features_ndarray, target_ndarray).
        """

        cols = self._inner_data.shape[-1]
        target = self._inner_data[:, self._target_column]

        idx = list(range(cols))
        idx.pop(self._target_column)

        data = self._inner_data[:, idx]

        return data, target

    def split_data(self):
        """
                Splits the loaded data into train/test sets using the provided data_splitter.

                This method updates `self.data_dict` by applying the splitting masks to
                every feature and target key in the dictionary.

                Raises:
                    AssertionError: If data_splitter is not defined or the splitter fails
                                    to provide 'train' and 'test' keys.
        """

        if not self.is_part_of_multisource:
            if not self._is_splitted:
                assert self.data_splitter is not None, "Data splitter must be set."

                masks = self.data_splitter(self.data_dict)
                assert all(v in masks.keys() for v in ['train', 'test']), f'Splitted data must have at least train and test values. {list(masks.keys())} given.'

                new_datadict = deepcopy(self.data_dict)

                for f, v in self.data_dict.items():
                    d = {}
                    for k, mask in masks.items():
                        d[k] = [v[mask]]

                    # new_datadict[k] = d
                    new_datadict.update({f : d})

                self.data_dict = new_datadict
                self._is_splitted = True

    def _load_data(self):
        """
                Executes the assigned loading function and constructs the initial data dictionary.

                Returns:
                    dict: The data dictionary containing 'features', the 'task_mode' target, and 'timestamps'.
        """

        data, target = self._loading_function()
        data_dict = {
            "features": data,
            self._task_mode: target,
            "timestamps": np.arange(len(data)),
        }

        self.data_dict = data_dict
        # self.split_data()
        self._is_loaded = True

        return self.data_dict

    @classmethod
    def load_from_csv(cls, csv: DataFrame, target_column: int | str, task_mode: str, data_splitter=None, **kwargs):
        """
                Alternative constructor to initialize the source specifically from a CSV DataFrame.

                Args:
                    csv (DataFrame): The source pandas DataFrame.
                    target_column (int | str): The column identifier for the target.
                    data_splitter (Callable, optional): Custom split logic.
                    **kwargs: Extra arguments for the class constructor.

                Returns:
                    CustomSingleSourceLoader: An instance configured for CSV loading.
        """

        return cls(csv, target_column, data_splitter=data_splitter, loading_type='csv', task_mode=task_mode, **kwargs)

    @classmethod
    def load_from_numpy(cls, numpy: np.ndarray, target_column: int | str, task_mode: str, data_splitter=None, **kwargs):
        """
                Alternative constructor to initialize the source specifically from a NumPy array.

                Args:
                    numpy (np.ndarray): The source NumPy array.
                    target_column (int | str): The index of the target column.
                    data_splitter (Callable, optional): Custom split logic.
                    **kwargs: Extra arguments for the class constructor.

                Returns:
                    CustomSingleSourceLoader: An instance configured for NumPy loading.
        """

        return cls(numpy, target_column, data_splitter=data_splitter, loading_type='numpy', task_mode=task_mode, **kwargs)

    def get_data(self) -> dict | list[dict]:
        """
                Retrieves the processed data wrapped in a SplitDatasetContainer, ready to be used as datasource.

                Returns:
                    SplitDatasetContainer: The container containing the split and loaded data.

                Raises:
                    AssertionError: If the data has not been loaded or split yet.
        """

        assert self._is_loaded is True, "Data must be loaded before getting data."
        assert self._is_splitted is True, "Data must be splitted before getting data."
        return SplitDatasetContainer(**self.data_dict)
