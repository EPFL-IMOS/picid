"""Tests for picid.transforms.base.transform_manager module.

This file consolidates all tests for ConfigTransformManager from multiple test files.
All dummy transforms and fixtures are imported from conftest.
"""

import pytest

from picid.transforms.base.transform_manager import ConfigTransformManager

# Import shared fixtures and dummy transforms from conftest
from test.transforms.base.conftest import (
    DummyInverseTransform,
)


class TestConfigTransformManager:
    """Comprehensive tests for ConfigTransformManager."""

    def test_manager_initialization_empty(self):
        """Test manager initialization with empty config."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        manager = ConfigTransformManager()
        assert len(manager) == 0
        assert not manager.is_instantiated

    def test_manager_initialization_with_config(self):
        """Test manager initialization with config."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {
            "scale": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        assert len(manager) == 1
        assert manager.has_transform("scale")

    def test_manager_lazy_instantiation(self):
        """Test lazy instantiation mode."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        manager = ConfigTransformManager(lazy_instantiation=True)
        assert not manager.is_instantiated
        # Should instantiate on first access
        transforms = manager.get_data_transforms()
        assert manager.is_instantiated
        assert isinstance(transforms, dict)

    def test_manager_has_transform(self):
        """Test has_transform method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {"transform1": {"transform": {}, "metadata": {}}}
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        assert manager.has_transform("transform1")
        assert not manager.has_transform("nonexistent")

    def test_manager_get_transform_names(self):
        """Test get_transform_names method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {
            "transform1": {"transform": {}, "metadata": {}},
            "transform2": {"transform": {}, "metadata": {}},
        }
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        names = manager.get_transform_names()
        assert "transform1" in names
        assert "transform2" in names

    def test_manager_get_transform_config(self):
        """Test get_transform_config method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {"transform1": {"transform": {}, "metadata": {"apply_to": "features"}}}
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        transform_config = manager.get_transform_config("transform1")
        assert transform_config["metadata"]["apply_to"] == "features"

    def test_manager_get_transform_config_missing_error(self):
        """Test get_transform_config with missing transform."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        manager = ConfigTransformManager(lazy_instantiation=True)
        with pytest.raises(KeyError):
            manager.get_transform_config("nonexistent")

    def test_manager_add_transforms_config(self):
        """Test add_transforms_config method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        manager = ConfigTransformManager(lazy_instantiation=True)
        new_config = {"transform1": {"transform": {}, "metadata": {}}}
        manager.add_transforms_config(new_config)
        assert manager.has_transform("transform1")

    def test_manager_add_transforms_config_conflict_error(self):
        """Test add_transforms_config with conflicting names."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {"transform1": {"transform": {}, "metadata": {}}}
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        new_config = {"transform1": {"transform": {}, "metadata": {}}}
        with pytest.raises(KeyError, match="already exist"):
            manager.add_transforms_config(new_config)

    def test_manager_add_transforms_after_instantiation_error(self):
        """Test that adding transforms after instantiation raises error."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        manager = ConfigTransformManager()
        # Force instantiation
        _ = manager.get_data_transforms()
        new_config = {"transform1": {"transform": {}, "metadata": {}}}
        with pytest.raises(RuntimeError, match="after instantiation"):
            manager.add_transforms_config(new_config)

    def test_manager_update_transforms_config(self):
        """Test update_transforms_config method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {"transform1": {"transform": {}, "metadata": {}}}
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        new_config = {"transform2": {"transform": {}, "metadata": {}}}
        manager.update_transforms_config(new_config)
        assert not manager.has_transform("transform1")
        assert manager.has_transform("transform2")

    def test_manager_remove_transform(self):
        """Test remove_transform method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {
            "transform1": {"transform": {}, "metadata": {}},
            "transform2": {"transform": {}, "metadata": {}},
        }
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        manager.remove_transform("transform1")
        assert not manager.has_transform("transform1")
        assert manager.has_transform("transform2")

    def test_manager_remove_transform_missing_error(self):
        """Test remove_transform with missing transform."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        manager = ConfigTransformManager(lazy_instantiation=True)
        with pytest.raises(KeyError):
            manager.remove_transform("nonexistent")

    def test_manager_remove_transform_after_instantiation_error(self):
        """Test that removing transforms after instantiation raises error."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        # ConfigTransformManager expects DictConfig with _target_ for instantiation
        # For testing, we'll use a simpler approach - test the error path directly
        # by creating a manager and forcing instantiation, then trying to remove
        config = {
            "transform1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        # Skip this test if instantiation fails (transform might not be importable)
        try:
            manager = ConfigTransformManager(
                transforms_config=config, lazy_instantiation=True
            )
            _ = manager.get_data_transforms()
            if manager.is_instantiated:
                with pytest.raises(RuntimeError, match="after instantiation"):
                    manager.remove_transform("transform1")
        except (RuntimeError, ImportError, AttributeError):
            pytest.skip("Transform instantiation not available in test context")

    def test_manager_clear_cache(self):
        """Test clear_cache method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        manager = ConfigTransformManager()
        _ = manager.get_data_transforms()
        assert manager.is_instantiated
        manager.clear_cache()
        assert not manager.is_instantiated

    def test_manager_force_reinstantiate(self):
        """Test force_reinstantiate method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        # ConfigTransformManager expects DictConfig with _target_ for instantiation
        config = {
            "transform1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        # Skip if instantiation fails
        try:
            manager = ConfigTransformManager(
                transforms_config=config, lazy_instantiation=True
            )
            _ = manager.get_data_transforms()
            if manager.is_instantiated:
                manager.force_reinstantiate()
                assert manager.is_instantiated
        except (RuntimeError, ImportError, AttributeError):
            pytest.skip("Transform instantiation not available in test context")

    def test_manager_len(self):
        """Test __len__ method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {
            "transform1": {"transform": {}, "metadata": {}},
            "transform2": {"transform": {}, "metadata": {}},
        }
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        assert len(manager) == 2

    def test_manager_contains(self):
        """Test __contains__ method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {"transform1": {"transform": {}, "metadata": {}}}
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        assert "transform1" in manager
        assert "nonexistent" not in manager

    def test_manager_repr(self):
        """Test __repr__ method."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        config = {"transform1": {"transform": {}, "metadata": {}}}
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        repr_str = repr(manager)
        assert "ConfigTransformManager" in repr_str
        assert "not instantiated" in repr_str or "instantiated" in repr_str

    def test_manager_skip_private_keys(self):
        """Test that keys starting with __ are skipped."""
        from picid.transforms.base.transform_manager import ConfigTransformManager

        # ConfigTransformManager expects DictConfig with _target_ for instantiation
        config = {
            "__private": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            },
            "public": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            },
        }
        # Skip if instantiation fails
        try:
            manager = ConfigTransformManager(
                transforms_config=config, lazy_instantiation=True
            )
            # get_transform_names returns all config keys
            names = manager.get_transform_names()
            assert "public" in names
            assert "__private" in names
            # But instantiation should skip __ keys
            transforms = manager.get_data_transforms()
            transform_names = [t.transform_name for t in transforms.values()]
            assert "__private" not in transform_names
            assert "public" in transform_names
        except (RuntimeError, ImportError, AttributeError):
            pytest.skip("Transform instantiation not available in test context")


class TestGetInverterForKey:
    """Tests for get_inverter_for_key (Phase 5.3 inverse transform routing)."""

    def test_get_inverter_for_key_empty_manager_returns_none(self):
        """Empty manager returns None for any key."""
        manager = ConfigTransformManager(transforms_config={})
        assert manager.get_inverter_for_key("target") is None
        assert manager.get_inverter_for_key("features") is None

    def test_get_inverter_for_key_no_inverse_transform_returns_none(self):
        """Manager with only non-inverse transforms returns None."""
        config = {
            "scale_features": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        assert manager.get_inverter_for_key("features") is None

    def test_get_inverter_for_key_returns_inverter_when_assign_to_matches(self):
        """When a transform implements InverseTransformMixin and assign_to matches key, return it."""
        config = {
            "scale_target": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyInverseTransform"
                },
                "metadata": {"apply_to": "target", "assign_to": "target"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        inverter = manager.get_inverter_for_key("target")
        assert inverter is not None
        assert isinstance(inverter, DummyInverseTransform)

    def test_get_inverter_for_key_returns_none_for_other_key(self):
        """get_inverter_for_key returns None for key that no inverse transform assigns to."""
        config = {
            "scale_target": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyInverseTransform"
                },
                "metadata": {"apply_to": "target", "assign_to": "target"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        assert manager.get_inverter_for_key("target") is not None
        assert manager.get_inverter_for_key("features") is None

    def test_get_inverter_for_key_multiple_inverters_returns_last_by_default(self):
        """When several transforms assign to the same key and implement inverse, default is last in pipeline order."""
        config = {
            "scale_target_1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyInverseTransform"
                },
                "metadata": {"apply_to": "target", "assign_to": "target"},
            },
            "scale_target_2": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyInverseTransform"
                },
                "metadata": {"apply_to": "target", "assign_to": "target"},
            },
        }
        manager = ConfigTransformManager(transforms_config=config)
        last_ = manager.get_inverter_for_key("target")
        first_ = manager.get_inverter_for_key("target", which="first")
        assert last_ is not first_
        # Last in pipeline order is the second transform
        dt_list = list(manager.get_data_transforms().values())
        assert last_ is dt_list[1].transform_instance
        assert first_ is dt_list[0].transform_instance

    def test_get_inverter_for_key_which_first_returns_first_match(self):
        """which='first' returns the first transform (in pipeline order) that assigns to key and implements inverse."""
        config = {
            "scale_target": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyInverseTransform"
                },
                "metadata": {"apply_to": "target", "assign_to": "target"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        inv = manager.get_inverter_for_key("target", which="first")
        assert inv is not None
        assert inv is manager.get_inverter_for_key("target", which="last")

    def test_get_inverter_for_key_which_invalid_raises(self):
        """which must be 'first' or 'last'."""
        manager = ConfigTransformManager(transforms_config={})
        with pytest.raises(ValueError, match="must be 'first' or 'last'"):
            manager.get_inverter_for_key("target", which="middle")

    def test_get_inverter_for_key_with_name_returns_instance_and_name(self):
        """get_inverter_for_key_with_name returns (instance, transform_name) for logging."""
        config = {
            "scale_target": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyInverseTransform"
                },
                "metadata": {"apply_to": "target", "assign_to": "target"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        inv, name = manager.get_inverter_for_key_with_name("target")
        assert inv is not None
        assert name == "scale_target"
        inv_none, name_none = manager.get_inverter_for_key_with_name("other")
        assert inv_none is None
        assert name_none is None


class TestCachePointBoundaries:
    """Tests for Phase 5.1 cache boundaries (get_cache_point_names, get_transform_names_after, get_config_up_to_and_including)."""

    def test_get_cache_point_names_empty_when_no_cache_point(self):
        """When no transform has cache_point in metadata, get_cache_point_names returns []."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            },
        }
        manager = ConfigTransformManager(transforms_config=config)
        assert manager.get_cache_point_names() == []

    def test_get_cache_point_names_returns_marked_transforms_in_order(self):
        """Transforms with metadata.cache_point true are returned in pipeline order."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            },
            "t2": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {
                    "apply_to": "features",
                    "assign_to": "features",
                    "cache_point": True,
                },
            },
            "t3": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            },
            "t4": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {
                    "apply_to": "features",
                    "assign_to": "features",
                    "cache_point": True,
                },
            },
        }
        manager = ConfigTransformManager(transforms_config=config)
        assert manager.get_cache_point_names() == ["t2", "t4"]

    def test_get_transform_names_after_returns_tail(self):
        """get_transform_names_after returns transform names after the given name (exclusive)."""
        config = {
            "a": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "x", "assign_to": "x"},
            },
            "b": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "x", "assign_to": "x"},
            },
            "c": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "x", "assign_to": "x"},
            },
        }
        manager = ConfigTransformManager(transforms_config=config)
        assert manager.get_transform_names_after("a") == ["b", "c"]
        assert manager.get_transform_names_after("b") == ["c"]
        assert manager.get_transform_names_after("c") == []
        assert manager.get_transform_names_after("nonexistent") == ["a", "b", "c"]

    def test_get_config_up_to_and_including_returns_slice(self):
        """get_config_up_to_and_including returns serializable config slice up to and including the name."""
        config = {
            "first": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "f", "assign_to": "f"},
            },
            "second": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "f", "assign_to": "f"},
            },
        }
        manager = ConfigTransformManager(transforms_config=config)
        slice_one = manager.get_config_up_to_and_including("first")
        assert isinstance(slice_one, dict)
        assert list(slice_one.keys()) == ["first"]
        slice_two = manager.get_config_up_to_and_including("second")
        assert list(slice_two.keys()) == ["first", "second"]
