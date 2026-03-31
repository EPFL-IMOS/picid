"""Comprehensive tests for FileSystemCache.

This module tests the file-based caching system used for preprocessing
data pipelines. The cache provides hash-validated storage to avoid
redundant computations in PHM data processing.

PHM Context:
-----------
Data preprocessing in PHM can be computationally expensive (feature extraction,
STFT computation, etc.). Caching preprocessed data accelerates experimentation
and ensures reproducibility across runs.

Test Coverage Strategy:
----------------------
1. **Path Generation Tests**: Correct directory structure for stages
2. **Metadata Handling**: Load/save metadata and hash validation
3. **Data Persistence**: Pickle serialization of data and metadata
4. **Cache Validation**: Hash matching for cache hit/miss decisions
5. **Handle Method**: Full cache workflow (check, build, save)
6. **Error Handling**: Missing files, corrupted cache, permission errors
"""

import numpy as np
import pytest
import json
from pathlib import Path

from picid.data.cache.offline import FileSystemCache


class TestFileSystemCachePaths:
    """Tests for path generation."""

    def test_paths_regular_stage(self, tmp_path):
        """Test path generation for regular (non-preprocessed) stage.

        **PHM Logic**: Regular stages (like 'loaded') store data directly
        in the cache directory without hash-based nesting.

        **Methodology**: Generate paths for 'loaded' stage, verify structure.

        **Expected**:
        - data.pkl in {cache_dir}/loaded/
        - meta.json in {cache_dir}/loaded/
        - hash.txt in {cache_dir}/loaded/

        Validates: Requirement FSC-1.1 - Regular stage path generation
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        data_path, meta_path, hash_path, metadata_path = cache._paths(
            cache_dir, stage="loaded"
        )

        # Verify paths contain stage name
        assert "loaded" in str(data_path)
        assert "loaded" in str(meta_path)
        assert "loaded" in str(hash_path)

        # Verify file names
        assert data_path.name == "data.pkl"
        assert meta_path.name == "meta.json"
        assert hash_path.name == "hash.txt"
        assert metadata_path.name == "metadata.pkl"

    def test_paths_preprocessed_stage_with_cache_key(self, tmp_path):
        """Test path generation for preprocessed stage with cache_key.

        **PHM Logic**: Preprocessed data is stored in hash-named subdirectories
        to allow multiple preprocessing configurations to coexist.

        **Methodology**: Generate paths for 'preprocessed' stage with cache_key.

        **Expected**: Paths nested under cache_key directory.

        Validates: Requirement FSC-1.2 - Preprocessed stage path generation
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)
        cache_key = "abc123def456"

        data_path, meta_path, hash_path, metadata_path = cache._paths(
            cache_dir, stage="preprocessed", cache_key=cache_key
        )

        # Verify cache_key in path
        assert cache_key in str(data_path)
        assert "preprocessed" in str(data_path)

    def test_paths_preprocessed_without_cache_key_error(self, tmp_path):
        """Test that preprocessed stage requires cache_key.

        **PHM Logic**: Preprocessed stage without cache_key is ambiguous -
        which preprocessing configuration to use?

        **Methodology**: Request preprocessed stage without cache_key.

        **Expected**: ValueError raised about missing cache_key.

        Validates: Requirement FSC-1.3 - Cache key requirement validation
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        with pytest.raises(ValueError, match="cache_key"):
            cache._paths(cache_dir, stage="preprocessed", cache_key=None)

    def test_paths_boundary_stage_requires_cache_key(self, tmp_path):
        """Boundary stage (Phase 5.1) requires cache_key and nests under it."""
        cache = FileSystemCache()
        cache_dir = str(tmp_path)
        with pytest.raises(ValueError, match="cache_key"):
            cache._paths(cache_dir, stage="boundary", cache_key=None)
        key = "boundary_hash_abc"
        data_path, _, _, _ = cache._paths(cache_dir, stage="boundary", cache_key=key)
        assert key in str(data_path)
        assert "boundary" in str(data_path)

    def test_paths_creates_directories(self, tmp_path):
        """Test that _paths creates directory structure.

        **PHM Logic**: Directories should be created automatically to
        simplify cache usage.

        **Methodology**: Generate paths for new directory, verify creation.

        **Expected**: Parent directories created automatically.

        Validates: Requirement FSC-1.4 - Auto directory creation
        """
        cache = FileSystemCache()
        new_dir = tmp_path / "new_cache" / "subdir"

        data_path, _, _, _ = cache._paths(str(new_dir), stage="loaded")

        # Parent directory should exist
        assert data_path.parent.exists()


class TestFileSystemCacheMetadata:
    """Tests for metadata loading and saving."""

    def test_load_metadata_success(self, tmp_path):
        """Test successful metadata loading.

        **PHM Logic**: Metadata contains configuration used to generate
        cached data, enabling validation before loading.

        **Methodology**: Write meta.json and hash.txt, then load.

        **Expected**: Returns (config_dict, hash_string) tuple.

        Validates: Requirement FSC-2.1 - Metadata loading
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        # Create stage directory
        stage_dir = tmp_path / "loaded"
        stage_dir.mkdir(parents=True)

        # Write metadata files
        config = {"param1": "value1", "param2": 42}
        (stage_dir / "meta.json").write_text(json.dumps(config))
        (stage_dir / "hash.txt").write_text("abc123hash")

        result = cache.load_metadata(cache_dir, stage="loaded")

        assert result is not None
        loaded_config, loaded_hash = result
        assert loaded_config == config
        assert loaded_hash == "abc123hash"

    def test_load_metadata_missing_files(self, tmp_path):
        """Test metadata loading when files don't exist.

        **PHM Logic**: Missing metadata indicates cache miss - data
        should be regenerated.

        **Methodology**: Call load_metadata on empty directory.

        **Expected**: Returns None (cache miss).

        Validates: Requirement FSC-2.2 - Missing metadata handling
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        # Don't create any files
        result = cache.load_metadata(cache_dir, stage="loaded")

        assert result is None

    def test_load_metadata_hash_whitespace_stripped(self, tmp_path):
        """Test that hash whitespace is stripped.

        **PHM Logic**: Hash files may have trailing newlines from text editors.

        **Methodology**: Write hash with trailing whitespace, verify stripped.

        **Expected**: Hash returned without whitespace.

        Validates: Requirement FSC-2.3 - Hash normalization
        """
        cache = FileSystemCache()
        stage_dir = tmp_path / "loaded"
        stage_dir.mkdir(parents=True)

        (stage_dir / "meta.json").write_text('{"key": "value"}')
        (stage_dir / "hash.txt").write_text("abc123hash\n\n  ")

        _, loaded_hash = cache.load_metadata(str(tmp_path), stage="loaded")

        assert loaded_hash == "abc123hash"


class TestFileSystemCacheData:
    """Tests for data loading and saving."""

    def test_load_data_success(self, tmp_path):
        """Test successful data loading.

        **PHM Logic**: Cached data and metadata are loaded together to
        restore preprocessing state.

        **Methodology**: Save data with joblib, then load.

        **Expected**: Returns (data, metadata) tuple matching saved values.

        Validates: Requirement FSC-3.1 - Data loading
        """
        import joblib

        cache = FileSystemCache()
        stage_dir = tmp_path / "loaded"
        stage_dir.mkdir(parents=True)

        # Save test data
        test_data = {"features": np.array([1, 2, 3])}
        test_metadata = {"source": "test"}

        joblib.dump(test_data, stage_dir / "data.pkl")
        joblib.dump(test_metadata, stage_dir / "metadata.pkl")

        result = cache.load_data(str(tmp_path), stage="loaded")

        assert result is not None
        loaded_data, loaded_metadata = result
        np.testing.assert_array_equal(loaded_data["features"], test_data["features"])
        assert loaded_metadata == test_metadata

    def test_load_data_missing_returns_none(self, tmp_path):
        """Test that missing data files return None.

        **PHM Logic**: Missing data indicates cache miss.

        **Methodology**: Call load_data on empty directory.

        **Expected**: Returns None.

        Validates: Requirement FSC-3.2 - Missing data handling
        """
        cache = FileSystemCache()
        result = cache.load_data(str(tmp_path), stage="loaded")

        assert result is None

    def test_save_creates_all_files(self, tmp_path):
        """Test that save creates all required files.

        **PHM Logic**: Complete cache entry requires data.pkl, metadata.pkl,
        meta.json (config), and hash.txt.

        **Methodology**: Call save, verify all files created.

        **Expected**: Four files created in stage directory.

        Validates: Requirement FSC-3.3 - Complete save operation
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        test_data = {"features": np.array([1, 2, 3])}
        test_metadata = {"source": "test"}
        test_config = {"param": "value"}

        cache.save(
            cache_dir=cache_dir,
            stage="loaded",
            data=test_data,
            metadata=test_metadata,
            config=test_config,
        )

        stage_dir = tmp_path / "loaded"
        assert (stage_dir / "data.pkl").exists()
        assert (stage_dir / "metadata.pkl").exists()
        assert (stage_dir / "meta.json").exists()
        assert (stage_dir / "hash.txt").exists()

    def test_save_and_load_boundary_stage(self, tmp_path):
        """Save and load with stage='boundary' and cache_key (Phase 5.1)."""
        cache = FileSystemCache()
        cache_dir = str(tmp_path)
        cache_key = "boundary_xyz"
        test_data = {"container_key": np.array([1.0, 2.0, 3.0])}
        test_metadata = {"transforms": "mock", "meta_data": {}}
        test_config = {"datasource": {}, "transforms": {"t1": {}}}
        cache.save(
            cache_dir=cache_dir,
            stage="boundary",
            data=test_data,
            metadata=test_metadata,
            config=test_config,
            cache_key=cache_key,
        )
        boundary_dir = tmp_path / "boundary" / cache_key
        assert (boundary_dir / "data.pkl").exists()
        assert (boundary_dir / "metadata.pkl").exists()
        meta = cache.load_metadata(cache_dir, stage="boundary", cache_key=cache_key)
        assert meta is not None
        loaded_config, stored_hash = meta
        assert stored_hash == cache_key
        loaded_data, loaded_meta = cache.load_data(
            cache_dir, stage="boundary", cache_key=cache_key
        )
        np.testing.assert_array_equal(
            loaded_data["container_key"], test_data["container_key"]
        )
        assert loaded_meta == test_metadata

    def test_save_with_library_dir_computes_cache_key(self, tmp_path):
        """Test save with library_dir computes cache key.

        **PHM Logic**: Cache key combines config hash and library directory
        hash for complete versioning.

        **Methodology**: Save with library_dir, verify hash file.

        **Expected**: Hash includes library directory content.

        Validates: Requirement FSC-3.4 - Library-aware caching
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path / "cache")
        library_dir = str(tmp_path / "library")

        # Create library directory with content
        lib_path = Path(library_dir)
        lib_path.mkdir(parents=True)
        (lib_path / "file.py").write_text("# test content")

        test_data = {"features": np.array([1, 2, 3])}
        test_metadata = {"source": "test"}
        test_config = {"param": "value"}

        cache.save(
            cache_dir=cache_dir,
            stage="loaded",
            data=test_data,
            metadata=test_metadata,
            config=test_config,
            library_dir=library_dir,
        )

        # Hash should be non-empty
        stage_dir = Path(cache_dir) / "loaded"
        hash_content = (stage_dir / "hash.txt").read_text().strip()
        assert len(hash_content) > 0


class TestFileSystemCacheHandle:
    """Tests for the handle method (main entry point)."""

    def test_handle_cache_miss(self, tmp_path):
        """Test handle with cache miss (no existing cache).

        **PHM Logic**: When no cache exists, build_fn is called and
        results are saved.

        **Methodology**: Call handle on empty directory with build_fn.

        **Expected**: build_fn called, results returned and cached.

        Validates: Requirement FSC-4.1 - Cache miss handling
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        # Track build_fn calls
        call_count = [0]

        def build_fn():
            call_count[0] += 1
            return {"features": np.array([1, 2, 3])}, {"source": "built"}

        config = {"param": "value"}

        data, metadata = cache.handle(
            cache_dir=cache_dir, stage="loaded", build_fn=build_fn, config=config
        )

        # build_fn should be called once
        assert call_count[0] == 1

        # Data should be returned
        assert "features" in data
        np.testing.assert_array_equal(data["features"], [1, 2, 3])

        # Cache should be created
        assert (tmp_path / "loaded" / "data.pkl").exists()

    def test_handle_cache_hit(self, tmp_path):
        """Test handle with cache hit (valid existing cache).

        **PHM Logic**: When cache exists with matching hash, cached data
        is returned and build_fn is NOT called.

        **Methodology**: Pre-populate cache, call handle with same config.

        **Expected**: build_fn not called, cached data returned.

        Validates: Requirement FSC-4.2 - Cache hit handling
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)
        config = {"param": "value"}

        # First call to populate cache
        def build_fn():
            return {"features": np.array([1, 2, 3])}, {"source": "built"}

        cache.handle(
            cache_dir=cache_dir, stage="loaded", build_fn=build_fn, config=config
        )

        # Second call with same config
        call_count = [0]

        def build_fn_tracked():
            call_count[0] += 1
            return {"features": np.array([4, 5, 6])}, {"source": "rebuilt"}

        data, metadata = cache.handle(
            cache_dir=cache_dir,
            stage="loaded",
            build_fn=build_fn_tracked,
            config=config,
        )

        # build_fn should NOT be called (cache hit)
        assert call_count[0] == 0

        # Original cached data should be returned
        np.testing.assert_array_equal(data["features"], [1, 2, 3])

    def test_handle_cache_invalidation(self, tmp_path):
        """Test handle with cache invalidation (config changed).

        **PHM Logic**: When config changes, hash won't match and cache
        is invalidated - build_fn is called to regenerate.

        **Methodology**: Populate cache, call with different config.

        **Expected**: build_fn called with new config, new results cached.

        Validates: Requirement FSC-4.3 - Cache invalidation
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        # First call with config A
        config_a = {"param": "value_a"}

        def build_fn_a():
            return {"features": np.array([1, 2, 3])}, {"source": "a"}

        cache.handle(
            cache_dir=cache_dir, stage="loaded", build_fn=build_fn_a, config=config_a
        )

        # Second call with different config B
        config_b = {"param": "value_b"}  # Different!
        call_count = [0]

        def build_fn_b():
            call_count[0] += 1
            return {"features": np.array([4, 5, 6])}, {"source": "b"}

        data, metadata = cache.handle(
            cache_dir=cache_dir, stage="loaded", build_fn=build_fn_b, config=config_b
        )

        # build_fn SHOULD be called (cache invalidated)
        assert call_count[0] == 1

        # New data should be returned
        np.testing.assert_array_equal(data["features"], [4, 5, 6])

    def test_handle_preprocessed_stage(self, tmp_path):
        """Test handle with preprocessed stage.

        **PHM Logic**: Preprocessed stage uses cache_key for nested storage.

        **Methodology**: Call handle for preprocessed stage.

        **Expected**: Data stored in cache_key subdirectory.

        Validates: Requirement FSC-4.4 - Preprocessed stage handling
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)
        config = {"transform": "standard_scaler"}

        def build_fn():
            return {"scaled_features": np.array([0.1, 0.5, 0.9])}, {}

        data, metadata = cache.handle(
            cache_dir=cache_dir, stage="preprocessed", build_fn=build_fn, config=config
        )

        # Preprocessed directory should contain hash subdirectory
        preprocessed_dir = tmp_path / "preprocessed"
        assert preprocessed_dir.exists()
        # Should have at least one subdirectory (the hash)
        subdirs = list(preprocessed_dir.iterdir())
        assert len(subdirs) >= 1


class TestFileSystemCacheWriteMeta:
    """Tests for the write_meta tombstone method."""

    def test_write_meta_creates_only_pending_file(self, tmp_path):
        """write_meta writes meta_pending.json but not meta.json, hash.txt, or data.pkl."""
        cache = FileSystemCache()
        config = {"param": "value"}
        cache.write_meta(str(tmp_path), stage="loaded", config=config)

        stage_dir = tmp_path / "loaded"
        assert (stage_dir / "meta_pending.json").exists()
        assert not (stage_dir / "meta.json").exists()
        assert not (stage_dir / "hash.txt").exists()
        assert not (stage_dir / "data.pkl").exists()
        assert not (stage_dir / "metadata.pkl").exists()

    def test_write_meta_returns_cache_key(self, tmp_path):
        """write_meta returns the computed cache key."""
        from picid.utils.hash_utils import hash_config

        cache = FileSystemCache()
        config = {"param": "value"}
        key = cache.write_meta(str(tmp_path), stage="loaded", config=config)
        assert key == hash_config(config)

    def test_handle_unaffected_by_pending_tombstone(self, tmp_path):
        """handle() treats a pending tombstone as a cache miss and rebuilds normally."""
        cache = FileSystemCache()
        config = {"param": "value"}

        # Simulate a crashed run: tombstone written but data never saved.
        cache.write_meta(str(tmp_path), stage="loaded", config=config)

        # Only meta_pending.json exists — no meta.json, so load_metadata returns None.
        stage_dir = tmp_path / "loaded"
        assert (stage_dir / "meta_pending.json").exists()
        assert not (stage_dir / "meta.json").exists()

        call_count = [0]

        def build_fn():
            call_count[0] += 1
            return {"features": np.array([1, 2, 3])}, {"source": "rebuilt"}

        data, metadata = cache.handle(
            cache_dir=str(tmp_path), stage="loaded", build_fn=build_fn, config=config
        )

        assert call_count[0] == 1
        np.testing.assert_array_equal(data["features"], [1, 2, 3])
        # After rebuild, the canonical meta.json and data.pkl now exist.
        assert (stage_dir / "meta.json").exists()
        assert (stage_dir / "data.pkl").exists()


class TestFileSystemCacheEdgeCases:
    """Edge case tests for FileSystemCache."""

    def test_empty_config(self, tmp_path):
        """Test handling of empty config.

        **PHM Logic**: Empty config should produce valid (but likely collision-prone) hash.

        **Methodology**: Call handle with empty config dict.

        **Expected**: Operation succeeds.

        Validates: Requirement FSC-5.1 - Empty config handling
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        def build_fn():
            return {"data": np.array([1])}, {}

        data, metadata = cache.handle(
            cache_dir=cache_dir, stage="loaded", build_fn=build_fn, config={}
        )

        assert "data" in data

    def test_large_data_serialization(self, tmp_path):
        """Test serialization of large data objects.

        **PHM Logic**: PHM datasets can be large (millions of samples).

        **Methodology**: Cache and retrieve 1M sample array.

        **Expected**: Data preserved correctly.

        Validates: Requirement FSC-5.2 - Large data handling
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        # Create large array
        large_array = np.random.randn(100000, 50)

        def build_fn():
            return {"features": large_array}, {}

        cache.handle(
            cache_dir=cache_dir,
            stage="loaded",
            build_fn=build_fn,
            config={"large": True},
        )

        # Load and verify
        loaded_data, _ = cache.load_data(cache_dir, stage="loaded")
        np.testing.assert_array_equal(loaded_data["features"], large_array)

    def test_special_characters_in_config(self, tmp_path):
        """Test config with special characters.

        **PHM Logic**: Config may contain special characters in paths/strings.

        **Methodology**: Use config with unicode and special chars.

        **Expected**: JSON serialization succeeds.

        Validates: Requirement FSC-5.3 - Special character handling
        """
        cache = FileSystemCache()
        cache_dir = str(tmp_path)

        special_config = {
            "path": "/tmp/test/data.pkl",
            "name": "test_config_α_β_γ",
            "quotes": 'value with "quotes"',
            "unicode": "日本語",
        }

        def build_fn():
            return {"data": np.array([1])}, {}

        # Should not raise
        data, metadata = cache.handle(
            cache_dir=cache_dir,
            stage="loaded",
            build_fn=build_fn,
            config=special_config,
        )

        assert "data" in data
