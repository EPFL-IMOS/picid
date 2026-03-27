"""Pytest configuration and fixtures for preprocessing tests.

Provides mock data loaders, transform managers, and sample data for testing
the preprocessing pipeline components.

Imports from picid.data.data_objects.data are deferred to inside fixtures
so that collection does not pull in awkward/pyarrow before plugins are ready.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock


# =============================================================================
# MOCK DATA STRUCTURES
# =============================================================================


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for splitting tests.

    Returns 1000 samples of 5 features with timestamps.
    """
    n_samples = 1000
    n_features = 5

    features = np.random.randn(n_samples, n_features)
    timestamps = np.arange(n_samples)
    target = np.random.randn(n_samples, 1)

    return {
        "features": features,
        "timestamps": timestamps,
        "target": target,
        "n_samples": n_samples,
        "n_features": n_features,
    }


@pytest.fixture
def small_time_series_data():
    """Create small time series data for edge case tests.

    Returns 100 samples - minimum viable for some splitters.
    """
    n_samples = 100
    n_features = 3

    return {
        "features": np.random.randn(n_samples, n_features),
        "timestamps": np.arange(n_samples),
        "target": np.random.randn(n_samples, 1),
        "n_samples": n_samples,
    }


@pytest.fixture
def sample_dataset_container():
    """Create a sample SplitDatasetContainer for testing."""
    from picid.data.data_objects import SplitDatasetContainer

    return SplitDatasetContainer(
        features={
            "train": [np.random.randn(100, 5)],
            "val": [np.random.randn(50, 5)],
            "test": [np.random.randn(50, 5)],
        },
        target={
            "train": [np.random.randn(100, 1)],
            "val": [np.random.randn(50, 1)],
            "test": [np.random.randn(50, 1)],
        },
    )


@pytest.fixture
def multi_unit_dataset_container():
    """Create a multi-unit SplitDatasetContainer."""
    from picid.data.data_objects import SplitDatasetContainer

    n_units = 3
    return SplitDatasetContainer(
        features={
            "train": [np.random.randn(100, 5) for _ in range(n_units)],
            "val": [np.random.randn(50, 5) for _ in range(n_units)],
            "test": [np.random.randn(50, 5) for _ in range(n_units)],
        },
        target={
            "train": [np.random.randn(100, 1) for _ in range(n_units)],
            "val": [np.random.randn(50, 1) for _ in range(n_units)],
            "test": [np.random.randn(50, 1) for _ in range(n_units)],
        },
    )


# =============================================================================
# MOCK LOADERS
# =============================================================================


@pytest.fixture
def mock_single_source_loader(sample_dataset_container):
    """Create a mock SingleSourceLoader."""
    loader = MagicMock()
    loader.get_data.return_value = sample_dataset_container
    loader.source_name = "test_source"
    return loader


@pytest.fixture
def mock_multi_source_loader(multi_unit_dataset_container):
    """Create a mock MultiSourceLoader."""
    loader = MagicMock()
    loader.get_data.return_value = multi_unit_dataset_container
    loader.source_names = ["unit_1", "unit_2", "unit_3"]
    return loader


# =============================================================================
# MOCK TRANSFORMS
# =============================================================================


class _StubTransformSequence:
    """Minimal stub satisfying TransformSequenceProtocol for PreProcessor tests."""

    def __init__(self):
        self.transforms = {}
        self._config = {}

    @property
    def config(self):
        return self._config

    def get_transforms(self):
        from collections import OrderedDict

        return OrderedDict(self.transforms)

    def get_cache_point_names(self):
        return []

    def get_transform_names_after(self, transform_name: str):
        return []

    def get_config_up_to_and_including(self, transform_name: str):
        return {}


class MockTransform:
    """Mock transform that multiplies by a factor."""

    def __init__(self, factor: float = 2.0):
        self.factor = factor
        self.fitted = False

    def fit_data(self, data, metadata):
        self.fitted = True
        return self

    def transform_data(self, data, metadata):
        key = list(data.keys())[0]
        return data[key] * self.factor


class MockDataTransform:
    """Mock DataTransform wrapper."""

    def __init__(self, transform=None, name="mock_transform"):
        self.transform = transform or MockTransform()
        self.name = name

    def __call__(self, data):
        return data


@pytest.fixture
def mock_transform_manager():
    """Create a stub satisfying TransformSequenceProtocol for PreProcessor tests."""
    return _StubTransformSequence()


@pytest.fixture
def mock_transform_manager_with_transforms():
    """Create a mock ConfigTransformManager with actual transforms."""
    manager = MagicMock()
    manager.transforms = {
        "scaler": MockDataTransform(name="scaler"),
        "normalizer": MockDataTransform(name="normalizer"),
    }
    return manager


# =============================================================================
# TIMESTAMP DATA
# =============================================================================


@pytest.fixture
def timestamp_data():
    """Create data with pandas timestamps for TimeStampSplitter tests."""
    import pandas as pd

    n_samples = 365 * 2  # 2 years of daily data
    start_date = "2020-01-01"

    dates = pd.date_range(start=start_date, periods=n_samples, freq="D")
    features = np.random.randn(n_samples, 5)
    target = np.random.randn(n_samples, 1)

    return {
        "features": features,
        "timestamps": pd.Series(dates),
        "target": target,
        "n_samples": n_samples,
        "start_date": start_date,
        "test_start": "2021-01-01",  # Split at year boundary
        "test_end": "2021-06-01",
    }


@pytest.fixture
def hourly_timestamp_data():
    """Create hourly timestamp data for fine-grained tests."""
    import pandas as pd

    n_samples = 24 * 30  # 30 days of hourly data
    start_date = "2020-01-01"

    dates = pd.date_range(start=start_date, periods=n_samples, freq="H")
    features = np.random.randn(n_samples, 3)

    return {
        "features": features,
        "timestamps": pd.Series(dates),
        "n_samples": n_samples,
    }


# =============================================================================
# MULTI-SOURCE DATA FOR BY_SOURCE_SPLITTER
# =============================================================================


@pytest.fixture
def multi_source_data():
    """Create multiple source data for BySourceSplitter tests."""
    from picid.data.data_objects import SplitDatasetContainer

    sources = {
        "source_a": SplitDatasetContainer(
            features={"all": [np.random.randn(100, 5)]},
            target={"all": [np.random.randn(100, 1)]},
        ),
        "source_b": SplitDatasetContainer(
            features={"all": [np.random.randn(80, 5)]},
            target={"all": [np.random.randn(80, 1)]},
        ),
        "source_c": SplitDatasetContainer(
            features={"all": [np.random.randn(120, 5)]},
            target={"all": [np.random.randn(120, 1)]},
        ),
    }
    return sources


@pytest.fixture
def source_containers():
    """Create list of DatasetContainers for source splitting."""
    from picid.data.data_objects import SplitDatasetContainer

    containers = [
        SplitDatasetContainer(
            features={"all": [np.random.randn(100, 5)]},
            target={"all": [np.random.randn(100, 1)]},
        )
        for _ in range(5)
    ]
    return containers
