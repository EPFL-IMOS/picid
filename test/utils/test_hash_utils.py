"""Comprehensive tests for hash_utils module.

This module tests the hashing utilities used for cache key generation
and configuration validation in the PHM data pipeline.

PHM Context:
-----------
Reproducible caching requires deterministic hashing of configurations
and library code. Hash changes when config or code changes, invalidating
stale cache entries.

Test Coverage Strategy:
----------------------
1. **Serialization Tests**: Converting configs to JSON-serializable format
2. **Config Hashing**: Deterministic SHA256 hashing of configurations
3. **Directory Hashing**: Hashing library directories for code versioning
4. **Cache Key Computation**: Combined config + library hashing
5. **Edge Cases**: Empty configs, nested structures, special types
"""

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf

from picid.utils.hash_utils import (
    ensure_serializable,
    hash_config,
    hash_directory,
    compute_cache_key,
)


class TestEnsureSerializable:
    """Tests for ensure_serializable function."""

    def test_dict_passthrough(self):
        """Test that regular dicts pass through correctly.

        **PHM Logic**: Regular Python dicts should be returned as-is for
        JSON serialization.

        **Methodology**: Pass a regular dict, verify unchanged.

        **Expected**: Same dict structure returned.

        Validates: Requirement HU-1.1 - Dict handling
        """
        test_dict = {"key": "value", "number": 42}
        result = ensure_serializable(test_dict)

        assert result == test_dict
        assert isinstance(result, dict)

    def test_dictconfig_conversion(self):
        """Test DictConfig to dict conversion.

        **PHM Logic**: Hydra configs (DictConfig) must be converted to
        regular dicts for JSON serialization.

        **Methodology**: Create DictConfig, verify converted to dict.

        **Expected**: Regular dict with same values.

        Validates: Requirement HU-1.2 - DictConfig conversion
        """
        cfg = OmegaConf.create({"key": "value", "number": 42})
        assert isinstance(cfg, DictConfig)

        result = ensure_serializable(cfg)

        assert isinstance(result, dict)
        assert not isinstance(result, DictConfig)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_listconfig_conversion(self):
        """Test ListConfig to list conversion.

        **PHM Logic**: Hydra list configs must be converted to regular lists.

        **Methodology**: Create ListConfig, verify converted to list.

        **Expected**: Regular list with same values.

        Validates: Requirement HU-1.3 - ListConfig conversion
        """
        cfg = OmegaConf.create([1, 2, 3, "four"])
        assert isinstance(cfg, ListConfig)

        result = ensure_serializable(cfg)

        assert isinstance(result, list)
        assert not isinstance(result, ListConfig)
        assert result == [1, 2, 3, "four"]

    def test_nested_config_conversion(self):
        """Test nested DictConfig/ListConfig conversion.

        **PHM Logic**: Deeply nested configs must be fully converted.

        **Methodology**: Create nested config, verify all levels converted.

        **Expected**: Fully converted nested structure.

        Validates: Requirement HU-1.4 - Nested config conversion
        """
        cfg = OmegaConf.create(
            {
                "level1": {"level2": {"items": [1, 2, {"nested": "value"}]}},
                "list_at_top": [{"a": 1}, {"b": 2}],
            }
        )

        result = ensure_serializable(cfg)

        # Verify all levels are regular Python types
        assert isinstance(result, dict)
        assert isinstance(result["level1"], dict)
        assert isinstance(result["level1"]["level2"], dict)
        assert isinstance(result["level1"]["level2"]["items"], list)
        assert isinstance(result["list_at_top"], list)
        assert isinstance(result["list_at_top"][0], dict)

    def test_set_to_sorted_list(self):
        """Test set conversion to sorted list.

        **PHM Logic**: Sets are not JSON-serializable, so they're converted
        to sorted lists for deterministic hashing.

        **Methodology**: Pass a set, verify converted to sorted list.

        **Expected**: Sorted list with same elements.

        Validates: Requirement HU-1.5 - Set conversion
        """
        test_set = {3, 1, 4, 1, 5, 9, 2, 6}  # Note: duplicates removed
        result = ensure_serializable(test_set)

        assert isinstance(result, list)
        assert result == sorted(test_set)  # Should be [1, 2, 3, 4, 5, 6, 9]

    def test_primitives_passthrough(self):
        """Test that primitive types pass through unchanged.

        **PHM Logic**: str, int, float, bool, None are JSON-native.

        **Methodology**: Pass primitives, verify unchanged.

        **Expected**: Same values returned.

        Validates: Requirement HU-1.6 - Primitive handling
        """
        assert ensure_serializable("string") == "string"
        assert ensure_serializable(42) == 42
        assert ensure_serializable(3.14) == 3.14
        assert ensure_serializable(True) is True
        assert ensure_serializable(None) is None

    def test_non_serializable_fallback(self):
        """Test fallback for non-serializable objects.

        **PHM Logic**: Unknown types fall back to str() representation.

        **Methodology**: Pass a custom object, verify string conversion.

        **Expected**: String representation of object.

        Validates: Requirement HU-1.7 - Fallback handling
        """

        class CustomObject:
            def __str__(self):
                return "CustomObject()"

        obj = CustomObject()
        result = ensure_serializable(obj)

        # Should fall back to string representation
        assert isinstance(result, str)
        assert "CustomObject" in result


class TestHashConfig:
    """Tests for hash_config function."""

    def test_hash_deterministic(self):
        """Test that hashing is deterministic.

        **PHM Logic**: Same config must produce same hash for cache matching.

        **Methodology**: Hash same config twice, verify identical hashes.

        **Expected**: Identical hash strings.

        Validates: Requirement HU-2.1 - Deterministic hashing
        """
        config = {"param1": "value1", "param2": 42}

        hash1 = hash_config(config)
        hash2 = hash_config(config)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest length

    def test_different_configs_different_hashes(self):
        """Test that different configs produce different hashes.

        **PHM Logic**: Config changes must invalidate cache (different hash).

        **Methodology**: Hash two different configs, verify different hashes.

        **Expected**: Different hash strings.

        Validates: Requirement HU-2.2 - Hash uniqueness
        """
        config1 = {"param": "value1"}
        config2 = {"param": "value2"}

        hash1 = hash_config(config1)
        hash2 = hash_config(config2)

        assert hash1 != hash2

    def test_hash_dictconfig(self):
        """Test hashing of DictConfig objects.

        **PHM Logic**: Hydra configs should hash identically to equivalent dicts.

        **Methodology**: Hash DictConfig and equivalent dict, compare.

        **Expected**: Same hash for equivalent structures.

        Validates: Requirement HU-2.3 - DictConfig hashing
        """
        regular_dict = {"key": "value", "number": 42}
        dict_config = OmegaConf.create(regular_dict)

        hash_dict = hash_config(regular_dict)
        hash_cfg = hash_config(dict_config)

        assert hash_dict == hash_cfg

    def test_hash_list(self):
        """Test hashing of list objects.

        **PHM Logic**: Lists should produce consistent hashes.

        **Methodology**: Hash a list, verify valid hash.

        **Expected**: Valid SHA256 hash string.

        Validates: Requirement HU-2.4 - List hashing
        """
        test_list = [1, 2, 3, {"nested": "dict"}]

        result = hash_config(test_list)

        assert len(result) == 64  # SHA256 hex digest

    def test_hash_empty_config(self):
        """Test hashing of empty config.

        **PHM Logic**: Empty configs should produce valid (consistent) hash.

        **Methodology**: Hash empty dict, verify valid hash.

        **Expected**: Valid SHA256 hash string.

        Validates: Requirement HU-2.5 - Empty config hashing
        """
        result = hash_config({})

        assert len(result) == 64
        # Hash should be deterministic even for empty
        assert hash_config({}) == hash_config({})

    def test_hash_key_order_independent(self):
        """Test that dict key order doesn't affect hash.

        **PHM Logic**: Python dicts maintain insertion order, but logically
        equivalent configs should hash identically.

        **Methodology**: Create dicts with different key orders, hash both.

        **Expected**: Same hash (assuming json.dumps with sort_keys=True).

        Validates: Requirement HU-2.6 - Key order independence
        """
        config1 = {"a": 1, "b": 2, "c": 3}
        config2 = {"c": 3, "a": 1, "b": 2}  # Different order

        hash1 = hash_config(config1)
        hash2 = hash_config(config2)

        # Should be equal if sorting is applied
        assert hash1 == hash2


class TestHashDirectory:
    """Tests for hash_directory function."""

    def test_hash_directory_basic(self, tmp_path):
        """Test basic directory hashing.

        **PHM Logic**: Library code changes should produce different hashes.

        **Methodology**: Create directory with files, compute hash.

        **Expected**: Valid SHA256 hash string.

        Validates: Requirement HU-3.1 - Basic directory hashing
        """
        # Create test files
        (tmp_path / "file1.py").write_text("def func(): pass")
        (tmp_path / "file2.py").write_text("class Test: pass")

        result = hash_directory(str(tmp_path))

        assert len(result) == 64  # SHA256 hex digest

    def test_hash_directory_deterministic(self, tmp_path):
        """Test that directory hashing is deterministic.

        **PHM Logic**: Same directory content = same hash.

        **Methodology**: Hash directory twice, verify identical.

        **Expected**: Identical hashes.

        Validates: Requirement HU-3.2 - Deterministic directory hashing
        """
        (tmp_path / "file.py").write_text("content")

        hash1 = hash_directory(str(tmp_path))
        hash2 = hash_directory(str(tmp_path))

        assert hash1 == hash2

    def test_hash_directory_content_change(self, tmp_path):
        """Test that content changes produce different hashes.

        **PHM Logic**: Code changes must invalidate cache.

        **Methodology**: Hash directory, modify file, hash again.

        **Expected**: Different hashes.

        Validates: Requirement HU-3.3 - Content-sensitive hashing
        """
        file_path = tmp_path / "file.py"
        file_path.write_text("version 1")

        hash1 = hash_directory(str(tmp_path))

        file_path.write_text("version 2")

        hash2 = hash_directory(str(tmp_path))

        assert hash1 != hash2

    def test_hash_directory_with_extension_filter(self, tmp_path):
        """Test directory hashing with extension filter.

        **PHM Logic**: Only relevant files (e.g., .py) should affect hash.

        **Methodology**: Create mixed files, hash with .py filter.

        **Expected**: Only .py files affect hash.

        Validates: Requirement HU-3.4 - Extension filtering
        """
        (tmp_path / "code.py").write_text("def func(): pass")
        (tmp_path / "data.csv").write_text("1,2,3")
        (tmp_path / "readme.txt").write_text("notes")

        hash_py_only = hash_directory(str(tmp_path), extensions=[".py"])

        # Modify non-.py file
        (tmp_path / "data.csv").write_text("4,5,6")

        hash_after_csv_change = hash_directory(str(tmp_path), extensions=[".py"])

        # Hash should be unchanged (only .py files matter)
        assert hash_py_only == hash_after_csv_change

    def test_hash_directory_nested(self, tmp_path):
        """Test hashing of nested directories.

        **PHM Logic**: All files in subdirectories should be included.

        **Methodology**: Create nested structure, verify files are hashed.

        **Expected**: Nested file changes affect hash.

        Validates: Requirement HU-3.5 - Recursive directory hashing
        """
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "top.py").write_text("top level")
        (subdir / "nested.py").write_text("nested file")

        hash1 = hash_directory(str(tmp_path))

        # Modify nested file
        (subdir / "nested.py").write_text("modified nested")

        hash2 = hash_directory(str(tmp_path))

        assert hash1 != hash2

    def test_hash_directory_empty(self, tmp_path):
        """Test hashing of empty directory.

        **PHM Logic**: Empty directories should produce valid hash.

        **Methodology**: Hash empty directory.

        **Expected**: Valid SHA256 hash (of empty content).

        Validates: Requirement HU-3.6 - Empty directory handling
        """
        result = hash_directory(str(tmp_path))

        assert len(result) == 64


class TestComputeCacheKey:
    """Tests for compute_cache_key function."""

    def test_compute_cache_key_basic(self, tmp_path):
        """Test basic cache key computation.

        **PHM Logic**: Cache key combines config hash and library hash.

        **Methodology**: Compute cache key with config and directory.

        **Expected**: Valid SHA256 hash string.

        Validates: Requirement HU-4.1 - Basic cache key computation
        """
        (tmp_path / "code.py").write_text("def func(): pass")
        config = {"param": "value"}

        result = compute_cache_key(config, str(tmp_path))

        assert len(result) == 64  # SHA256 hex digest

    def test_compute_cache_key_deterministic(self, tmp_path):
        """Test cache key determinism.

        **PHM Logic**: Same inputs = same cache key.

        **Methodology**: Compute cache key twice, verify identical.

        **Expected**: Identical keys.

        Validates: Requirement HU-4.2 - Deterministic cache key
        """
        (tmp_path / "code.py").write_text("content")
        config = {"param": "value"}

        key1 = compute_cache_key(config, str(tmp_path))
        key2 = compute_cache_key(config, str(tmp_path))

        assert key1 == key2

    def test_compute_cache_key_config_change(self, tmp_path):
        """Test cache key changes with config.

        **PHM Logic**: Config changes must produce different cache key.

        **Methodology**: Change config, verify different key.

        **Expected**: Different keys.

        Validates: Requirement HU-4.3 - Config-sensitive cache key
        """
        (tmp_path / "code.py").write_text("content")

        config1 = {"param": "value1"}
        config2 = {"param": "value2"}

        key1 = compute_cache_key(config1, str(tmp_path))
        key2 = compute_cache_key(config2, str(tmp_path))

        assert key1 != key2

    def test_compute_cache_key_library_change(self, tmp_path):
        """Test cache key changes with library.

        **PHM Logic**: Library code changes must produce different cache key.

        **Methodology**: Change library file, verify different key.

        **Expected**: Different keys.

        Validates: Requirement HU-4.4 - Library-sensitive cache key
        """
        file_path = tmp_path / "code.py"
        file_path.write_text("version 1")
        config = {"param": "value"}

        key1 = compute_cache_key(config, str(tmp_path))

        file_path.write_text("version 2")

        key2 = compute_cache_key(config, str(tmp_path))

        assert key1 != key2

    def test_compute_cache_key_with_path_object(self, tmp_path):
        """Test cache key with Path object.

        **PHM Logic**: Both str and Path objects should work.

        **Methodology**: Pass Path object as library_dir.

        **Expected**: Valid cache key.

        Validates: Requirement HU-4.5 - Path object support
        """
        (tmp_path / "code.py").write_text("content")
        config = {"param": "value"}

        # Pass Path object instead of string
        result = compute_cache_key(config, tmp_path)

        assert len(result) == 64

    def test_compute_cache_key_multiple_directories(self, tmp_path):
        """Test cache key with multiple library directories.

        **PHM Logic**: Multiple library dirs should all contribute to hash.

        **Methodology**: Pass list of directories.

        **Expected**: Valid cache key combining all directories.

        Validates: Requirement HU-4.6 - Multiple directory support
        """
        dir1 = tmp_path / "lib1"
        dir2 = tmp_path / "lib2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "code1.py").write_text("lib1 content")
        (dir2 / "code2.py").write_text("lib2 content")

        config = {"param": "value"}

        result = compute_cache_key(config, [str(dir1), str(dir2)])

        assert len(result) == 64

    def test_compute_cache_key_invalid_library_type(self):
        """Test cache key with invalid library_dir type.

        **PHM Logic**: Only str, Path, or list of those should be accepted.

        **Methodology**: Pass invalid type (int).

        **Expected**: TypeError raised.

        Validates: Requirement HU-4.7 - Type validation
        """
        config = {"param": "value"}

        with pytest.raises(TypeError):
            compute_cache_key(config, 12345)  # Invalid type
