import logging
from typing import Literal, Callable, Any, Union

import numpy as np
from pandas import DataFrame

from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.data.datasources.base.single_source_loader import SingleSourceLoader
from picid.data.preprocessing import BySourceSplitter

log = logging.getLogger(__name__)

class CustomSingleSourceLoader(SingleSourceLoader):
    """
        A data source loader that handles single-source data from either CSV (DataFrame)
        or NumPy formats, supporting custom splitting and task-specific formatting.

        Wraps raw tabular data (pandas DataFrame or NumPy ndarray) and provides a
        unified interface for loading, splitting, and exposing features and targets
        regardless of the underlying format. The appropriate internal loading function
        is selected at construction time based on `loading_type`.

        If `data` is already a dictionary with 'train', 'val', and 'test' keys, the
        loader treats the data as pre-split and skips the splitting step entirely.

        Args:
            data (DataFrame | np.ndarray | dict[str: DataFrame | np.ndarray]): Raw input data.
                May be a single array/DataFrame or a pre-split dictionary with 'train', 'val', and
                'test' keys.
            target_column (int | str): Column index or name identifying the target
                feature to predict. Negative indexes are allowed.
            task_mode (str): Task type string (e.g. 'classification', 'regression'),
                used as a key in the returned data dictionary.
            loading_type (Literal['csv', 'numpy']): Selects the internal loading
                strategy. Use 'csv' for pandas DataFrames and 'numpy' for ndarrays.
            is_part_of_multisource (bool): Whether this loader is a component of a
                larger MultiSourceLoader. Defaults to False.
            data_splitter (Callable[[dict], dict], optional): A function that partitions
                data into subsets. It receives a dictionary of all features and targets,
                and must return a mask for each split ('train', 'val', 'test').
            data_name (str): Human-readable label for this data source, used in logs.
            **kwargs: Additional arguments forwarded to SingleSourceLoader.

        Raises:
            AssertionError: If `data` is a dict but does not contain all three required
                split keys ('train', 'val', 'test').
            AssertionError: If `loading_type` is not 'csv' or 'numpy'.

        Example:
            >>> loader = CustomSingleSourceLoader.load_from_numpy(
            ...     source=my_array,
            ...     target_column=-1,
            ...     task_mode='classification'
            ... )
            >>> loader = CustomSingleSourceLoader.load_from_csv(
            ...     source=my_dataframe,
            ...     target_column='label',
            ...     task_mode='regression'
            ... )
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
        """Initialise the loader. See class docstring for parameter details."""

        super().__init__(
            is_part_of_multisource=is_part_of_multisource,
            data_splitter=data_splitter,
            data_name=data_name,
            task_mode=task_mode,
            **kwargs)

        if isinstance(data, dict):
            missing = [k for k in ['train', 'val', 'test'] if k not in data]
            if missing:
                raise ValueError(
                    f"Pre-split data must contain all three splits ['train', 'val', 'test'], "
                    f"but {list(data.keys())} were given (missing: {missing})."
                )
            self._is_splitted = True

        if loading_type == 'csv':
            self._loading_function = self._load_csv
        elif loading_type == 'numpy':
            if target_column < 0:
                if isinstance(data, dict):
                    target_column = np.arange(data['train'].shape[1])[target_column]
                else:
                    target_column = np.arange(data.shape[1])[target_column]
            self._loading_function = self._load_numpy
        else:
            raise ValueError(
                f'loading_type must be either "csv" or "numpy", got {loading_type!r}.'
            )

        self._inner_data = data
        self._target_column = target_column
        self._task_mode = task_mode

        log.info(f'Created {data_name} datasource from {loading_type} with target column {target_column}.')

    def _load_csv(self, split: str = None):
        """
        Extracts features and target from a pandas DataFrame.

        If the loader holds pre-split data, `split` selects the appropriate subset.
        The target column is dropped from the features before returning.

        Args:
            split (str, optional): One of 'train', 'val', or 'test'. When provided,
                selects the corresponding subset from a pre-split data dictionary.
                If None, the full unsplit DataFrame is used.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple of (features, target), both as
                NumPy arrays.
        """
        if split is not None:
            current_data = self._inner_data[split]
        else:
            current_data = self._inner_data

        if isinstance(self._target_column, str):
            df_target = current_data[self._target_column].to_frame().copy()
            data = current_data.drop(columns=[self._target_column], inplace=False)
        else:
            df_target = current_data.iloc[:, self._target_column].to_frame().copy()
            data = current_data.drop(current_data.columns[self._target_column], axis=1, inplace=False)

        return data.to_numpy(), df_target.to_numpy()

    def _load_numpy(self, split: str = None):
        """
        Extracts features and target from a NumPy array.

        If the loader holds pre-split data, `split` selects the appropriate subset.
        The target column is removed from the feature matrix before returning.

        Args:
            split (str, optional): One of 'train', 'val', or 'test'. When provided,
                selects the corresponding subset from a pre-split data dictionary.
                If None, the full unsplit array is used.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple of (features, target) arrays,
                where the target column has been removed from features.
        """
        if split is not None:
            current_data = self._inner_data[split]
        else:
            current_data = self._inner_data

        cols = current_data.shape[-1]
        target = current_data[:, self._target_column]

        idx = list(range(cols))
        idx.pop(self._target_column)

        data = current_data[:, idx]

        return data, target

    def _load_data(self):
        """
        Runs the configured loading function and assembles the internal data dictionary.

        If the raw data is pre-split (i.e. a dict), each split is loaded individually
        and the resulting features and targets are stored as nested dictionaries keyed
        by split name. Otherwise, the full dataset is loaded at once.

        The resulting data dictionary always contains:
            - 'features': the input feature matrix (array or dict of arrays per split).
            - task_mode key (e.g. 'classification'): the target array or dict of targets.
            - 'timestamps': a sequential integer index over the samples.

        Returns
        -------
        dict
            The assembled data dictionary with keys ``"features"``, the task-mode
            key (e.g. ``"rul"``), and ``"timestamps"``. Also stored as
            ``self.data_dict``.
        """
        stamps = {}
        if isinstance(self._inner_data, dict):
            dd = {}
            dt = {}
            for k, v in self._inner_data.items():
                a, b = self._loading_function(split=k)
                dd[k] = a
                dt[k] = b
                stamps[k] = np.arange(len(a))

            data = dd
            target = dt
        else:
            data, target = self._loading_function()
            stamps = np.arange(len(data))

        data_dict = {
            "features": data,
            self._task_mode: target,
            "timestamps": stamps,
        }

        self.data_dict = data_dict
        self._is_loaded = True

        if isinstance(self._inner_data, dict):
            log.info(
                f'Loaded {self.data_name} already split datasource resulting following split:')
            for k, v in data_dict['features'].items():
                # print()
                log.info(f'{k}: {v.shape}')

        else:
            log.info(f'Loaded {self.data_name} datasource resulting in data shape {data.shape} for task {self._task_mode} '
                     f'with target shape {target.shape}.')

        return self.data_dict

    @classmethod
    def load_from_csv(cls, source: Union[DataFrame, dict[str, DataFrame]], target_column: int | str, task_mode: str, data_splitter=None, **kwargs):
        """
        Alternative constructor that creates an instance configured for CSV/DataFrame input.

        Args:
            source (DataFrame | dict[str, DataFrame]): A pandas DataFrame or a pre-split
                dictionary of DataFrames keyed by split name ('train', 'val', 'test').
            target_column (int | str): Column name or integer index identifying the target.
            task_mode (str): Task type string forwarded to the constructor.
            data_splitter (Callable, optional): Custom split logic forwarded to the constructor.
            **kwargs: Additional arguments forwarded to the constructor.

        Returns:
            CustomSingleSourceLoader: An instance configured with `loading_type='csv'`.
        """

        log.info('Creating the single source loader from csv.')

        return cls(source, target_column, data_splitter=data_splitter, loading_type='csv', task_mode=task_mode, **kwargs)

    @classmethod
    def load_from_numpy(cls, source: Union[np.ndarray, dict[str, np.ndarray]], target_column: int | str, task_mode: str, data_splitter=None, **kwargs):
        """
        Alternative constructor that creates an instance configured for NumPy array input.

        Negative `target_column` indices are resolved at construction time against the
        array's column count, so downstream code always works with a positive index.

        Args:
            source (np.ndarray | dict[str, np.ndarray]): A NumPy array or a pre-split
                dictionary of arrays keyed by split name ('train', 'val', 'test').
            target_column (int): Integer index of the target column. Supports negative
                indexing (e.g. -1 for the last column).
            task_mode (str): Task type string forwarded to the constructor.
            data_splitter (Callable, optional): Custom split logic forwarded to the constructor.
            **kwargs: Additional arguments forwarded to the constructor.

        Returns:
            CustomSingleSourceLoader: An instance configured with `loading_type='numpy'`.
        """

        return cls(source, target_column, data_splitter=data_splitter, loading_type='numpy', task_mode=task_mode, **kwargs)


class CustomMultiSourceLoader(MultiSourceLoader):
    """
    A data source loader that combines multiple CustomSingleSourceLoader instances
    into a unified multi-source dataset, optionally applying a shared cross-source
    splitting strategy.

    Each source is identified by a string key and encapsulated in its own
    CustomSingleSourceLoader. Splitting can be handled either per-source (each
    SingleSourceLoader carries its own data_splitter, or it is already split) or globally across all sources
    via a BySourceSplitter passed as `multisource_data_splitter`. The two approaches
    are mutually exclusive: if a global splitter is provided, no individual source
    may have its own data_splitter.

    Args:
        sources (dict[str, CustomSingleSourceLoader]): Mapping from source name to its
            corresponding loader instance.
        task_mode (str): Task type string (e.g. 'classification', 'regression').
        multisource_data_splitter (BySourceSplitter | None): An optional splitter that
            operates across all sources simultaneously. Mutually exclusive with
            per-source data_splitter. Defaults to None.
        data_name (str): Human-readable label for this combined data source.
        **kwargs: Additional arguments forwarded to MultiSourceLoader.

    Raises:
        AssertionError: If `multisource_data_splitter` is provided but one or more
            individual sources also have a data_splitter set.

    Example:
        >>> loader = CustomMultiSourceLoader.load_from_primitive(
        ...     sources={'site_a': array_a, 'site_b': array_b},
        ...     target_column=-1,
        ...     task_mode='classification',
        ...     data_splitter=BySourceSplitter(...)
        ... )
    """

    def __init__(self,
                 sources: dict[str, CustomSingleSourceLoader],
                 task_mode: str,
                 multisource_data_splitter: BySourceSplitter | None = None,
                 data_name: str = 'custom multi source datasource',
                 **kwargs):

        """Initialise the loader. See class docstring for parameter details."""

        log.info(f'Creating Multi Source Loader (name: {data_name}) with task mode {task_mode} '
                 f'and sources dictionary: {list(sources.keys())}')

        super().__init__(source_list=sources, data_name=data_name, task_mode=task_mode,
                         multisource_data_splitter=multisource_data_splitter, **kwargs)

        if multisource_data_splitter:
            check = [s.data_splitter is None for s in self.data_source_dict.values()]
            assert all(check), \
                ('If multisource_data_splitter is not None, '
                 f'the sources cannot have data_splitter != None (given {check}).')

    @classmethod
    def load_from_primitive(cls, sources: dict[str, np.ndarray | DataFrame],
                            target_column: int | str, task_mode: str,
                            data_splitter: dict[str, Any] | None | BySourceSplitter = None,
                            **kwargs):
        """
        Alternative constructor that builds a CustomMultiSourceLoader directly from
        raw NumPy arrays or pandas DataFrames, creating the underlying
        CustomSingleSourceLoader instances automatically.

        The splitting behavior depends on the type of `data_splitter`:

        - **BySourceSplitter**: A single global splitter is applied across all sources.
          Individual loaders are created with `is_part_of_multisource=True` and no
          per-source splitter.
        - **dict**: A mapping from source name to a per-source splitter. Each loader
          receives its own splitter and `is_part_of_multisource=False`. The dict must
          contain exactly one entry per source.

        Args:
            sources (dict[str, np.ndarray | DataFrame]): Mapping from source name to its
                raw data. Each value may be a NumPy array or a pandas DataFrame.
            target_column (int | str): Column index or name shared across all sources
                that identifies the target feature.
            task_mode (str): Task type string forwarded to each SingleSourceLoader and
                to the MultiSourceLoader constructor.
            data_splitter (dict[str, Any] | None | BySourceSplitter): Splitting strategy.
                Pass a BySourceSplitter for global splitting or a dict for per-source
                splitting. When a dict is used its length must match the number of sources.
            **kwargs: Additional arguments forwarded to the class constructor.

        Raises:
            AssertionError: If `data_splitter` is a dict whose length does not match
                the number of sources.

        Returns:
            CustomMultiSourceLoader: A fully initialised multi-source loader.
        """

        _sources = []

        log.info(f'Creating a Multi Source Loader from primitives with task mode {task_mode}. Number of sources: {len(sources)}, '
                 f'target columns: {target_column}.')

        is_part_of_multisource = False
        if isinstance(data_splitter, BySourceSplitter):
            is_part_of_multisource = True
        elif data_splitter is not None:
            if not isinstance(data_splitter, dict):
                raise TypeError(
                    f"data_splitter must be a BySourceSplitter or a dict mapping source names to "
                    f"splitters, got {type(data_splitter)}."
                )
            if len(sources) != len(data_splitter):
                raise ValueError(
                    f"When data_splitter is a dict it must have one entry per source, "
                    f"but got {len(sources)} sources and {len(data_splitter)} splitters."
                )

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
