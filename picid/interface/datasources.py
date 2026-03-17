import logging
from typing import Literal, Callable, Any

import numpy as np
from pandas import DataFrame

from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.data.datasources.base.single_source_loader import SingleSourceLoader
from picid.data.preprocessing import BySourceSplitter

log = logging.getLogger(__name__)

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

        log.info(f'Created {data_name} datasource from {loading_type} with target column {target_column}.')

    def _load_csv(self):
        """
                Internal method to extract features and targets from a pandas DataFrame.

                Returns:
                    tuple: A tuple containing (features_dataframe, target_series).
        """

        if isinstance(self._target_column, str):
            df_target = self._inner_data[self._target_column].to_frame().copy()
            data = self._inner_data.drop(columns=[self._target_column], inplace=False)
        else:
            df_target = self._inner_data.iloc[:, self._target_column].to_frame().copy()
            data = self._inner_data.drop(self._inner_data.columns[self._target_column], axis=1, inplace=False)

        return data.to_numpy(), df_target.to_numpy()

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
        self._is_loaded = True

        log.info(f'Loaded {self.data_name} datasource resulting in data shape {data.shape} for task {self._task_mode} '
                 f'with target shape {target.shape}.')

        return self.data_dict

    @classmethod
    def load_from_csv(cls, source: DataFrame, target_column: int | str, task_mode: str, data_splitter=None, **kwargs):
        """
                Alternative constructor to initialize the source specifically from a CSV DataFrame.

                Args:
                    source (DataFrame): The source pandas DataFrame.
                    target_column (int | str): The column identifier for the target.
                    data_splitter (Callable, optional): Custom split logic.
                    **kwargs: Extra arguments for the class constructor.

                Returns:
                    CustomSingleSourceLoader: An instance configured for CSV loading.
        """

        log.info(f'Creating the single source loader from csv.')

        return cls(source, target_column, data_splitter=data_splitter, loading_type='csv', task_mode=task_mode, **kwargs)

    @classmethod
    def load_from_numpy(cls, source: np.ndarray, target_column: int | str, task_mode: str, data_splitter=None, **kwargs):
        """
                Alternative constructor to initialize the source specifically from a NumPy array.

                Args:
                    source (np.ndarray): The source NumPy array.
                    target_column (int | str): The index of the target column.
                    data_splitter (Callable, optional): Custom split logic.
                    **kwargs: Extra arguments for the class constructor.

                Returns:
                    CustomSingleSourceLoader: An instance configured for NumPy loading.
        """

        return cls(source, target_column, data_splitter=data_splitter, loading_type='numpy', task_mode=task_mode, **kwargs)


class CustomMultiSourceLoader(MultiSourceLoader):
    """
    A data source loader class designed to handle single-source data from either CSV (DataFrame)
    or NumPy formats, supporting custom splitting and task-specific formatting.
    """

    def __init__(self,
                 sources: dict[str, CustomSingleSourceLoader],
                 task_mode: str,
                 multisource_data_splitter: BySourceSplitter | None = None,
                 data_name: str = 'custom multi source datasource',
                 **kwargs):

        log.info(f'Creating Multi Source Loader (name: {data_name}) with task mode {task_mode} '
                 f'and sources dictionary: {list(sources.keys())}')

        super().__init__(source_list=sources, data_name=data_name, task_mode=task_mode,
                         multisource_data_splitter=multisource_data_splitter, **kwargs)

        if multisource_data_splitter:
            check = [s.data_splitter is None for s in self.data_source_dict.values()]
            assert all(check), \
                (f'If multisource_data_splitter is not None, '
                 f'the sources cannot have data_splitter != None (given {check}).')

    @classmethod
    def load_from_primitive(cls, sources: dict[str, np.ndarray | DataFrame],
                            target_column: int | str, task_mode: str,
                            data_splitter: dict[str, Any] | None | BySourceSplitter = None,
                            **kwargs):
        _sources = []

        log.info(f'Creating a Multi Source Loader from primitives with task mode {task_mode}. Number of sources: {len(sources)}, '
                 f'target columns: {target_column}.')

        is_part_of_multisource = False
        if isinstance(data_splitter, BySourceSplitter):
            is_part_of_multisource = True
        else:
            assert isinstance(sources, dict) and len(sources) == len(data_splitter), ('when is_part_of_multisource=None '
                                                                                      'the number of sources must be '
                                                                                      'equal to the number of '
                                                                                      'single_source_data_splitter.')

        _sources = {}
        for s, source in sources.items():

            ints = \
                {
                'source': source,
                'data_splitter': data_splitter.get(s) if not is_part_of_multisource else None,
                'target_column': target_column,
                'task_mode' : task_mode,
                'is_part_of_multisource' : is_part_of_multisource
                }

            if isinstance(source, np.ndarray):
                source = CustomSingleSourceLoader.load_from_numpy(**ints)
            elif isinstance(source, DataFrame):
                source = CustomSingleSourceLoader.load_from_csv(**ints)

            _sources[s] = source

        return cls(_sources,
                   target_column=target_column,
                   task_mode=task_mode,
                   multisource_data_splitter=data_splitter if is_part_of_multisource else None,
                   **kwargs)
