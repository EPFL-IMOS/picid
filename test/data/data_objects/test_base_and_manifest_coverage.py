"""
Coverage-focused tests for picid.data.data_objects.base and manifest.
No changes to picid/data allowed; tests only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picid.data.data_objects import (
    BaseDataObject,
    BaseDataObjectWithMetadata,
)
from picid.data.data_objects.manifest import ManifestEntry, MetadataManifest


# ---------------------------------------------------------------------------
# ToNumpyMixin / ToDataFrameMixin (via BaseDataObjectWithMetadata)
# ---------------------------------------------------------------------------


def test_to_numpy_key_not_found_raises():
    """to_numpy with key not in container raises KeyError (ToNumpyMixin)."""
    c = BaseDataObjectWithMetadata(features=np.zeros((5, 2)))
    with pytest.raises(KeyError, match="not found"):
        c.to_numpy("missing_key")


def test_to_dataframe_key_not_found_raises():
    """to_dataframe with key not in container raises KeyError (ToDataFrameMixin)."""
    c = BaseDataObjectWithMetadata(x=np.zeros((3, 2)))
    with pytest.raises(KeyError, match="not found"):
        c.to_dataframe("missing_key")


def test_to_dataframe_returns_value_when_already_dataframe():
    """to_dataframe when value is already a DataFrame returns it (no conversion)."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    c = BaseDataObjectWithMetadata(table=df)
    out = c.to_dataframe("table")
    assert out is df


def test_to_dataframe_uses_metadata_column_map():
    """to_dataframe uses metadata.column_map[key] for column names when present."""
    c = BaseDataObjectWithMetadata(values=np.array([[1, 2], [3, 4]]))
    c._metadata = {"column_map": {"values": ["x", "y"]}}
    df = c.to_dataframe("values")
    assert list(df.columns) == ["x", "y"]


# ---------------------------------------------------------------------------
# BaseDataObject (via subclass that doesn't add much)
# ---------------------------------------------------------------------------


class MinimalBaseDataObject(BaseDataObject):
    """Minimal subclass to exercise BaseDataObject without metadata."""

    pass


def test_base_data_object_setitem_empty_list_warns():
    """__setitem__ preserves empty lists so known empty splits are not lost."""
    obj = MinimalBaseDataObject()
    obj["k"] = []
    assert "k" in obj
    assert obj["k"] == []


def test_base_data_object_setitem_invalid_type_raises():
    """__setitem__ with disallowed type raises TypeError."""
    obj = MinimalBaseDataObject()
    with pytest.raises(TypeError, match="Value must be"):
        obj["k"] = "not allowed"


def test_base_data_object_delitem():
    """__delitem__ removes key."""
    obj = MinimalBaseDataObject(a=np.zeros(3))
    assert "a" in obj
    del obj["a"]
    assert "a" not in obj


def test_base_data_object_getattr_missing_raises_attribute_error():
    """__getattr__ for missing key raises AttributeError (not KeyError)."""
    obj = MinimalBaseDataObject(a=np.zeros(2))
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = obj.missing_attr


def test_base_data_object_repr_with_data():
    """__repr__ with non-empty _data shows key=value parts."""
    obj = MinimalBaseDataObject(x=np.array([1]))
    r = repr(obj)
    assert "MinimalBaseDataObject" in r
    assert "x=" in r


def test_base_data_object_copy_shallow():
    """copy(deep=False) does shallow copy of _data and _instance_cls."""
    obj = MinimalBaseDataObject(a=np.zeros(5))
    other = obj.copy(deep=False)
    assert other["a"] is obj["a"]
    assert other._data is not obj._data
    assert other._data["a"] is obj["a"]


# ---------------------------------------------------------------------------
# BaseDataObjectWithMetadata
# ---------------------------------------------------------------------------


def test_base_with_metadata_explicit_metadata_param():
    """BaseDataObjectWithMetadata(metadata=...) sets _metadata; column_map updated for DataFrames."""
    c = BaseDataObjectWithMetadata(
        metadata={"custom": "value"},
        values=np.array([[1, 2], [3, 4]]),
    )
    assert c._metadata is not None
    assert c._metadata.get("custom") == "value"
    assert "column_map" in c._metadata
    assert list(c.keys()) == ["values"]


def test_base_with_metadata_setitem_removes_stale_column_map():
    """Replacing a DataFrame key with non-DataFrame removes its column_map entry."""
    c = BaseDataObjectWithMetadata()
    c._metadata = {"column_map": {}}
    c["meta"] = pd.DataFrame({"a": [1]})
    assert "meta" in c._metadata.get("column_map", {})
    c["meta"] = np.zeros((1, 2))
    assert "meta" not in c._metadata.get("column_map", {})


def test_base_with_metadata_copy_deep_false_metadata():
    """copy(deep=False) copies metadata with .copy() (shallow)."""
    c = BaseDataObjectWithMetadata(x=np.zeros(2))
    c._metadata = {"col": [1, 2]}
    c2 = c.copy(deep=False)
    assert c2._metadata is not c._metadata
    assert c2._metadata == {"col": [1, 2]}
    c2._metadata["col"].append(3)
    assert len(c._metadata["col"]) == 3  # shared list


def test_base_with_metadata_copy_metadata_none():
    """copy when _metadata is None leaves new._metadata as None."""
    c = BaseDataObjectWithMetadata(x=np.zeros(1))
    c._metadata = None
    c2 = c.copy(deep=True)
    assert c2._metadata is None


def test_base_with_metadata_validate_with_keys_to_validate():
    """validate(keys_to_validate=[...]) only validates those keys."""
    c = BaseDataObjectWithMetadata(a=np.zeros(5), b=np.zeros(5))
    c.validate(keys_to_validate=["a", "b"])
    c["c"] = np.zeros(3)
    # Validating only a,b should not compare with c
    c.validate(keys_to_validate=["a", "b"])


# ---------------------------------------------------------------------------
# ManifestEntry
# ---------------------------------------------------------------------------


def test_manifest_entry_invalid_category_raises():
    """ManifestEntry with category not 'datasource' or 'transform' raises."""
    with pytest.raises(ValueError, match="category must be"):
        ManifestEntry(
            schema_version="1.0",
            producer_version="0.1",
            category="invalid",
            payload={},
        )


# ---------------------------------------------------------------------------
# MetadataManifest
# ---------------------------------------------------------------------------


def test_manifest_query_by_key():
    """query(key=...) returns only entries with that key."""
    m = MetadataManifest()
    m.add(
        ManifestEntry(
            schema_version="1.0",
            producer_version="0.1",
            category="transform",
            payload={},
            key="features",
        )
    )
    m.add(
        ManifestEntry(
            schema_version="1.0",
            producer_version="0.1",
            category="transform",
            payload={},
            key="target",
        )
    )
    q = m.query(key="features")
    assert len(q) == 1
    assert q[0].key == "features"


def test_manifest_query_by_split():
    """query(split=...) returns only entries with that split."""
    m = MetadataManifest()
    m.add(
        ManifestEntry(
            schema_version="1.0",
            producer_version="0.1",
            category="transform",
            payload={},
            split="train",
        )
    )
    m.add(
        ManifestEntry(
            schema_version="1.0",
            producer_version="0.1",
            category="transform",
            payload={},
            split="val",
        )
    )
    q = m.query(split="train")
    assert len(q) == 1
    assert q[0].split == "train"


def test_manifest_copy_deep_false():
    """copy(deep=False) uses dict(e.payload) for payloads."""
    m = MetadataManifest()
    m.add(
        ManifestEntry(
            schema_version="1.0",
            producer_version="0.1",
            category="transform",
            payload={"a": 1},
            step_id="X",
        )
    )
    m2 = m.copy(deep=False)
    assert len(m2) == 1
    assert m2._entries[0].payload["a"] == 1
