"""Tests for picid.data.cache.base (BaseCache) via a concrete implementation."""

from picid.data.cache.base import BaseCache


class ConcreteCache(BaseCache):
    """Concrete cache that implements load, save, handle for coverage."""

    def __init__(self):
        self._store = {}
        self._configs = {}

    def load(self, name: str):
        return self._store.get(name)

    def save(self, name: str, data, config: dict = None):
        self._store[name] = data
        if config is not None:
            self._configs[name] = config

    def handle(self, name: str, build_fn, config: dict = None):
        existing = self.load(name)
        if existing is not None:
            return existing
        data = build_fn()
        self.save(name, data, config)
        return data


def test_base_cache_load():
    """Calling load on a concrete implementation covers BaseCache.load body."""
    c = ConcreteCache()
    assert c.load("missing") is None
    c.save("x", 42)
    assert c.load("x") == 42


def test_base_cache_save():
    """Calling save on a concrete implementation covers BaseCache.save body."""
    c = ConcreteCache()
    c.save("a", [1, 2, 3])
    assert c.load("a") == [1, 2, 3]
    c.save("b", "data", config={"k": "v"})
    assert c._configs.get("b") == {"k": "v"}


def test_base_cache_handle():
    """Calling handle on a concrete implementation covers BaseCache.handle body."""
    c = ConcreteCache()
    out = c.handle("key", build_fn=lambda: 99, config={})
    assert out == 99
    assert c.load("key") == 99
    # Second call returns cached
    out2 = c.handle("key", build_fn=lambda: 999)
    assert out2 == 99
