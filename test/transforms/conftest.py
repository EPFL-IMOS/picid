"""Pytest configuration and fixtures for transform tests."""
import numpy as np
import pytest
import awkward as ak
from lmetk.data.data_objects.data import NamedTransformInput, SplitDatasetContainer


@pytest.fixture
def sample_2d_array():
    """Create a sample 2D numpy array for testing."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_1d_array():
    """Create a sample 1D numpy array for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_named_transform_input():
    """Create a sample NamedTransformInput for testing."""
    return NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


@pytest.fixture
def sample_multi_key_input():
    """Create a NamedTransformInput with multiple keys."""
    return NamedTransformInput(
        features=np.array([[1.0, 2.0], [3.0, 4.0]]),
        target=np.array([[0.5], [0.7]])
    )


@pytest.fixture
def sample_split_dataset_container():
    """Create a sample SplitDatasetContainer for integration tests."""
    return SplitDatasetContainer(
        features={
            "train": [np.array([[1.0, 2.0], [3.0, 4.0]])],
            "val": [np.array([[5.0, 6.0]])],
            "test": [np.array([[7.0, 8.0]])]
        },
        target={
            "train": [np.array([[0.1], [0.2]])],
            "val": [np.array([[0.3]])],
            "test": [np.array([[0.4]])]
        }
    )


@pytest.fixture
def sample_ragged_array():
    """Create a sample ragged (awkward) array for testing."""
    return ak.Array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ])


@pytest.fixture
def sample_time_series_signal():
    """Create a sample time series signal for spectral/time statistics tests."""
    n = 128
    t = np.linspace(0, 4 * np.pi, n)
    signal = np.sin(t)
    return signal.reshape(-1, 1)


@pytest.fixture
def sample_multi_signal_array():
    """Create a multi-signal array for testing."""
    n = 100
    signals = [
        np.sin(np.linspace(0, 4 * np.pi, n)),
        np.cos(np.linspace(0, 4 * np.pi, n)),
        np.sin(2 * np.linspace(0, 4 * np.pi, n))
    ]
    return np.column_stack(signals)
