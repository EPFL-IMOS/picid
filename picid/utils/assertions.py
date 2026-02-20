from omegaconf import ListConfig
import numpy as np


def assert_list_of_ndarray_or_nd_array_for_dims(input_data, dims):
    # Support for list, OmegaConf ListConfig, and np.ndarray

    if isinstance(input_data, (list, ListConfig)):
        for arr in input_data:
            assert isinstance(
                arr, np.ndarray
            ), f"Element is not a numpy ndarray. arr type: {type(arr)}"
            assert arr.ndim == dims, f"Array does not have {dims} dimensions."
    elif isinstance(input_data, np.ndarray):
        assert input_data.ndim == dims, f"Array does not have {dims} dimensions."
    else:
        raise AssertionError("Input is neither a list/ListConfig nor a numpy ndarray.")
