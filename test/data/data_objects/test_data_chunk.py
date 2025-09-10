import pytest
from lmetk.data.data_objects.data import DataChunk


def test_datachunk_metadata_set_and_get():
    metadata = {"source": "sensor1", "unit": "Celsius"}
    chunk = DataChunk(instance_cls=int, data={"a": 1}, metadata=metadata)
    assert chunk.metadata == metadata


def test_datachunk_metadata_update():
    chunk = DataChunk(instance_cls=int, data={"a": 1}, metadata={"source": "sensor1"})
    chunk.metadata["unit"] = "Celsius"
    assert chunk.metadata["unit"] == "Celsius"
    assert chunk.metadata["source"] == "sensor1"


def test_datachunk_metadata_empty_by_default():
    chunk = DataChunk(instance_cls=int, data={"a": 1})
    assert isinstance(chunk.metadata, dict)
    assert chunk.metadata == {}


def test_datachunk_metadata_is_independent_between_instances():
    chunk1 = DataChunk(instance_cls=int, data={"a": 1}, metadata={"a": 1})
    chunk2 = DataChunk(instance_cls=int, data={"b": 2}, metadata={"b": 2})
    chunk1.metadata["a"] = 10
    assert chunk2.metadata["b"] == 2
    assert chunk1.metadata["a"] == 10
