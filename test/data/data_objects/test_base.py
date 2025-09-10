import pytest

from lmetk.data.data_objects.base import BaseDataObject


class Dummy:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Dummy({self.value!r})"

    def __eq__(self, other):
        return isinstance(other, Dummy) and self.value == other.value

    def __hash__(self):
        return hash(self.value)


def test_init_and_getitem():
    obj = BaseDataObject(Dummy, a=Dummy(1), b=Dummy(2))
    assert obj["a"] == Dummy(1)
    assert obj["b"] == Dummy(2)
    assert len(obj) == 2


def test_setitem_and_contains():
    obj = BaseDataObject(Dummy)
    obj["x"] = Dummy(42)
    assert "x" in obj
    assert obj["x"] == Dummy(42)


def test_setitem_with_dict():
    obj = BaseDataObject(Dummy)
    obj["nested"] = {"foo": Dummy(5)}
    assert isinstance(obj["nested"], BaseDataObject)
    assert obj["nested"]["foo"] == Dummy(5)


def test_setitem_type_error():
    obj = BaseDataObject(Dummy)
    with pytest.raises(TypeError):
        obj["bad"] = 123  # not Dummy, not dict, not BaseDataObject


def test_delitem():
    obj = BaseDataObject(Dummy, a=Dummy(1))
    del obj["a"]
    assert "a" not in obj


def test_attribute_access():
    obj = BaseDataObject(Dummy, x=Dummy(10))
    assert obj.x == Dummy(10)
    with pytest.raises(AttributeError):
        _ = obj.y


def test_iter_and_keys_values_items():
    obj = BaseDataObject(Dummy, a=Dummy(1), b=Dummy(2))
    assert set(obj.keys()) == {"a", "b"}
    assert set(obj.values()) == {Dummy(1), Dummy(2)}
    assert set(obj.items()) == {("a", Dummy(1)), ("b", Dummy(2))}
    assert set(iter(obj)) == {"a", "b"}


def test_repr_empty_and_nonempty():
    obj = BaseDataObject(Dummy)
    assert repr(obj) == "BaseDataObject()"
    obj["foo"] = Dummy(7)
    assert "foo=Dummy(7)" in repr(obj)
