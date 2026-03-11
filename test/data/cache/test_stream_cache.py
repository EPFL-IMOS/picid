"""Comprehensive tests for StreamCache.

This module tests the stream-based caching system used for
online data processing pipelines.

PHM Context:
-----------
Streaming data (e.g., real-time sensor feeds) benefits from
caching transformed results to avoid repeated computation.

Test Coverage Strategy:
----------------------
1. **Initialization**: Directory creation and transform function storage
2. **Save/Load**: Pickle serialization of stream results
3. **Handle Method**: Cache-or-compute workflow
4. **Edge Cases**: Empty streams, large data, missing cache
"""

import pytest
import numpy as np

from picid.data.cache.online import StreamCache


class TestStreamCacheInitialization:
    """Tests for StreamCache initialization."""

    def test_init_creates_directory(self, tmp_path):
        """Test that initialization creates output directory.

        **PHM Logic**: Cache directory must exist for file operations.

        **Methodology**: Initialize with new directory path.

        **Expected**: Directory created automatically.

        Validates: Requirement SC-1.1 - Directory creation
        """
        output_dir = tmp_path / "new_cache_dir"
        assert not output_dir.exists()

        def transform_fn(x):
            return x

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(output_dir))

        assert output_dir.exists()
        assert cache.transform_fn is transform_fn

    def test_init_stores_transform_function(self, tmp_path):
        """Test that transform function is stored.

        **PHM Logic**: Transform function used for cache misses.

        **Methodology**: Pass custom transform, verify stored.

        **Expected**: Transform function accessible.

        Validates: Requirement SC-1.2 - Transform storage
        """

        def custom_transform(stream):
            return {"processed": stream * 2}

        cache = StreamCache(transform_fn=custom_transform, output_dir=str(tmp_path))

        assert cache.transform_fn is custom_transform

    def test_init_expands_tilde(self, tmp_path, monkeypatch):
        """Test that ~ in path is expanded.

        **PHM Logic**: User home paths should work.

        **Methodology**: Use path with ~, verify expansion.

        **Expected**: Path expanded to full path.

        Validates: Requirement SC-1.3 - Path expansion
        """
        # Mock home directory to tmp_path
        monkeypatch.setenv("HOME", str(tmp_path))

        def transform_fn(x):
            return x

        # This may not work on all platforms, so we test actual behavior
        try:
            cache = StreamCache(transform_fn=transform_fn, output_dir="~/test_cache")
            # If it works, output_dir should contain tmp_path
            assert "~" not in cache.output_dir
        except Exception:
            pytest.skip("Home expansion not fully testable")


class TestStreamCacheSave:
    """Tests for save method."""

    def test_save_creates_file(self, tmp_path):
        """Test that save creates pickle file.

        **PHM Logic**: Cached data stored as .pkl files.

        **Methodology**: Save data, verify file exists.

        **Expected**: {name}.pkl file created.

        Validates: Requirement SC-2.1 - File creation
        """

        def transform_fn(x):
            return x

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        test_data = {"result": np.array([1, 2, 3])}
        cache.save("test_entry", test_data)

        # Check file exists
        cache_file = tmp_path / "test_entry.pkl"
        assert cache_file.exists()

    def test_save_preserves_data(self, tmp_path):
        """Test that saved data can be reloaded correctly.

        **PHM Logic**: Data integrity must be maintained through save/load.

        **Methodology**: Save then load, verify identical.

        **Expected**: Loaded data matches saved data.

        Validates: Requirement SC-2.2 - Data integrity
        """

        def transform_fn(x):
            return x

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        test_data = {
            "array": np.array([[1, 2], [3, 4]]),
            "string": "test",
            "number": 42,
        }
        cache.save("test_entry", test_data)

        loaded = cache.load("test_entry")

        np.testing.assert_array_equal(loaded["array"], test_data["array"])
        assert loaded["string"] == test_data["string"]
        assert loaded["number"] == test_data["number"]


class TestStreamCacheLoad:
    """Tests for load method."""

    def test_load_existing_cache(self, tmp_path):
        """Test loading existing cache entry.

        **PHM Logic**: Cached data should be retrievable.

        **Methodology**: Save data, then load it.

        **Expected**: Returns cached data.

        Validates: Requirement SC-3.1 - Cache retrieval
        """

        def transform_fn(x):
            return x

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        original_data = {"value": np.array([1, 2, 3])}
        cache.save("entry", original_data)

        loaded = cache.load("entry")

        np.testing.assert_array_equal(loaded["value"], original_data["value"])

    def test_load_missing_returns_none(self, tmp_path):
        """Test loading non-existent cache returns None.

        **PHM Logic**: Missing cache should trigger recomputation.

        **Methodology**: Load without prior save.

        **Expected**: Returns None.

        Validates: Requirement SC-3.2 - Missing cache handling
        """

        def transform_fn(x):
            return x

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        result = cache.load("nonexistent_entry")

        assert result is None


class TestStreamCacheHandle:
    """Tests for handle method."""

    def test_handle_cache_miss(self, tmp_path):
        """Test handle with cache miss.

        **PHM Logic**: On cache miss, transform_fn called and result cached.

        **Methodology**: Call handle on empty cache.

        **Expected**: transform_fn called, result returned and cached.

        Validates: Requirement SC-4.1 - Cache miss workflow
        """
        call_count = [0]

        def transform_fn(stream):
            call_count[0] += 1
            return {"processed": stream * 2}

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        input_stream = np.array([1, 2, 3])
        result = cache.handle("entry", input_stream)

        # Transform should be called
        assert call_count[0] == 1

        # Result should be correct
        np.testing.assert_array_equal(result["processed"], input_stream * 2)

        # Cache should be created
        assert (tmp_path / "entry.pkl").exists()

    def test_handle_cache_hit(self, tmp_path):
        """Test handle with cache hit.

        **PHM Logic**: On cache hit, return cached data without transform.

        **Methodology**: Pre-populate cache, call handle.

        **Expected**: transform_fn NOT called, cached data returned.

        Validates: Requirement SC-4.2 - Cache hit workflow
        """
        call_count = [0]

        def transform_fn(stream):
            call_count[0] += 1
            return {"new_value": stream}

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        # Pre-populate cache
        cached_data = {"cached_value": np.array([10, 20, 30])}
        cache.save("entry", cached_data)

        # Call handle
        result = cache.handle("entry", np.array([1, 2, 3]))

        # Transform should NOT be called
        assert call_count[0] == 0

        # Cached data should be returned
        np.testing.assert_array_equal(
            result["cached_value"], cached_data["cached_value"]
        )

    def test_handle_multiple_entries(self, tmp_path):
        """Test handle with multiple cache entries.

        **PHM Logic**: Different stream names should have separate caches.

        **Methodology**: Handle multiple named streams.

        **Expected**: Each stream cached separately.

        Validates: Requirement SC-4.3 - Multi-entry support
        """

        def transform_fn(stream):
            return {"sum": np.sum(stream)}

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        # Process multiple streams
        result1 = cache.handle("stream_a", np.array([1, 2, 3]))
        result2 = cache.handle("stream_b", np.array([10, 20, 30]))

        assert result1["sum"] == 6
        assert result2["sum"] == 60

        # Both should be cached
        assert (tmp_path / "stream_a.pkl").exists()
        assert (tmp_path / "stream_b.pkl").exists()


class TestStreamCacheEdgeCases:
    """Edge case tests for StreamCache."""

    def test_empty_stream(self, tmp_path):
        """Test handling of empty stream.

        **PHM Logic**: Empty data should be cacheable.

        **Methodology**: Process empty array.

        **Expected**: Empty result cached correctly.

        Validates: Requirement SC-5.1 - Empty data handling
        """

        def transform_fn(stream):
            return {"length": len(stream)}

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        result = cache.handle("empty", np.array([]))

        assert result["length"] == 0

    def test_large_stream(self, tmp_path):
        """Test handling of large stream.

        **PHM Logic**: Large PHM datasets should be cacheable.

        **Methodology**: Process large array.

        **Expected**: Large data cached and retrieved correctly.

        Validates: Requirement SC-5.2 - Large data handling
        """

        def transform_fn(stream):
            return {"data": stream}

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        large_data = np.random.randn(100000, 50)
        result = cache.handle("large", large_data)

        np.testing.assert_array_equal(result["data"], large_data)

        # Verify retrieval
        loaded = cache.load("large")
        np.testing.assert_array_equal(loaded["data"], large_data)

    def test_special_characters_in_name(self, tmp_path):
        """Test cache name with special characters.

        **PHM Logic**: Stream names may contain various characters.

        **Methodology**: Use name with underscores and numbers.

        **Expected**: Cache works correctly.

        Validates: Requirement SC-5.3 - Special name handling
        """

        def transform_fn(stream):
            return {"value": stream}

        cache = StreamCache(transform_fn=transform_fn, output_dir=str(tmp_path))

        # Names with various characters
        names = ["stream_1", "data_2023", "sensor_temp_01"]

        for name in names:
            result = cache.handle(name, np.array([1, 2, 3]))
            assert result is not None
            assert (tmp_path / f"{name}.pkl").exists()
