import numpy as np
import pytest

from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.preprocessing import TimeSplitter
from picid.interface import CustomSingleSourceLoader



@pytest.mark.parametrize('get_sources', [[1, 'numpy'], [1, 'csv']], indirect=True)
def test_single_source_loader_standalone_load_split_get_returns_split_container(
    get_sources,
):
    """Standalone: load → split → get_data returns SplitDatasetContainer."""
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "target"],
    )

    source = get_sources[0]
    if isinstance(source, np.ndarray):
        loader = CustomSingleSourceLoader.load_from_numpy(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')
    else:
        loader = CustomSingleSourceLoader.load_from_csv(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')

    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data
    assert "target" in data
    assert "train" in data["features"]
    assert "val" in data["features"]
    assert "test" in data["features"]


@pytest.mark.parametrize('get_sources', [[1, 'numpy'], [1, 'csv']], indirect=True)
def test_single_source_loader_get_data_before_load_raises(get_sources):
    """get_data() before load_data() raises AssertionError."""
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "target"],
    )

    source = get_sources[0]
    if isinstance(source, np.ndarray):
        loader = CustomSingleSourceLoader.load_from_numpy(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')
    else:
        loader = CustomSingleSourceLoader.load_from_csv(source,
                                                        target_column=-1,
                                                        data_splitter=data_splitter,
                                                        task_mode='target')

    with pytest.raises(AssertionError, match="must be loaded"):
        loader.get_data()


@pytest.mark.parametrize('get_sources', [[1, 'numpy'], [1, 'csv']], indirect=True)
def test_single_source_loader_get_data_before_split_raises(get_sources):
    """Standalone: get_data() before split_data() raises AssertionError."""
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "target"],
    )

    source = get_sources[0]
    if isinstance(source, np.ndarray):
        loader = CustomSingleSourceLoader.load_from_numpy(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')
    else:
        loader = CustomSingleSourceLoader.load_from_csv(source,
                                                        target_column=-1,
                                                        data_splitter=data_splitter,
                                                        task_mode='target')

    loader.load_data()
    with pytest.raises(AssertionError, match="must be splitted"):
        loader.get_data()


@pytest.mark.parametrize('get_sources', [[1, 'numpy'], [1, 'csv']], indirect=True)
def test_single_source_loader_get_meta_data_after_load(get_sources):
    """get_meta_data() after load returns dict (empty for CustomSingleSourceLoader)."""
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "target"],
    )

    source = get_sources[0]
    if isinstance(source, np.ndarray):
        loader = CustomSingleSourceLoader.load_from_numpy(source,
                                                          target_column=-1,
                                                          data_splitter=data_splitter,
                                                          task_mode='target')
    else:
        loader = CustomSingleSourceLoader.load_from_csv(source,
                                                        target_column=-1,
                                                        data_splitter=data_splitter,
                                                        task_mode='target')

    loader.load_data()
    loader.load_data()
    meta = loader.get_meta_data()
    assert isinstance(meta, dict)