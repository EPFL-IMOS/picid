"""Pytest configuration and fixtures for cache tests."""

import numpy as np
import pytest


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def sample_cache_data():
    """Create sample data for cache tests."""
    return {
        "features": np.random.randn(100, 5),
        "target": np.random.randn(100, 1),
        "metadata": {"source": "test", "version": 1},
    }


@pytest.fixture
def sample_config():
    """Create sample configuration for cache tests."""
    return {
        "transform": "standard_scaler",
        "param1": "value1",
        "param2": 42,
        "nested": {"inner": "value"},
    }


@pytest.fixture
def library_dir_with_files(tmp_path):
    """Create library directory with Python files."""
    lib_dir = tmp_path / "library"
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Create some Python files
    (lib_dir / "module1.py").write_text("def func1(): pass")
    (lib_dir / "module2.py").write_text("class Class2: pass")
    (lib_dir / "utils" / "__init__.py").parent.mkdir(exist_ok=True)
    (lib_dir / "utils" / "__init__.py").write_text("# utils")
    (lib_dir / "utils" / "helpers.py").write_text("def helper(): pass")

    return lib_dir
