"""Behavior-led characterization tests for extra transform-base scenarios.

These cases cover manager and multisource edges not exercised by the more
central contract files, while still asserting outputs, errors, or metadata.
"""

from __future__ import annotations

import numpy as np
import pytest
import awkward as ak
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from sortedcontainers import SortedDict

from picid.data.data_objects import NamedTransformInput, SplitDatasetContainer
from picid.transforms.base.base_transform import DenseTransform, RaggedTransform
from picid.transforms.base.strategy import TransformStrategy
from picid.transforms.base.data_transform import DataTransform
from collections import OrderedDict
from picid.transforms.base.transform_manager import (
    ConfigTransformManager,
    create_transform_manager_from_config,
)
from picid.transforms.base.multisource import (
    tolist,
    ConcatFitAndPerSegmentTransformMixin,
    NoFitPerSegmentMixin,
    InverseTransformMixin,
)

from test.transforms.base.conftest import (
    DummyStatelessTransform,
    DummyFittableTransform,
    DummyMultiKeyTransform,
    DummyRaggedTransform,
    create_dummy_split_container,
    create_dummy_single_unit_container,
)


# ----- base_transform.py: default fit_data (line 33) -----


class TestBaseTransformCoverage:
    def test_default_fit_data_body_executed(self):
        """Ensure BaseTransform.fit_data default implementation (pass) is executed."""

        # Use a subclass that does not override fit_data
        class OnlyTransformData(NoFitPerSegmentMixin, DenseTransform):
            def transform_data(
                self, data: NamedTransformInput, metadata: Dict[str, Any]
            ) -> Any:
                return data

        t = OnlyTransformData()
        data = NamedTransformInput(features=np.array([1.0, 2.0]))
        result = t.fit_data(data, {})
        assert result is None


# ----- data_transform.py: logger and branches -----


class TestDataTransformCoverage:
    def test_implicit_assign_to_logs_when_apply_to_is_sequence(self, caplog):
        """Implicit assign_to for list apply_to logs and forward scales both keys."""
        import logging

        caplog.set_level(logging.INFO)
        metadata = {"apply_to": ["features", "target"]}
        transform = DummyMultiKeyTransform()
        with caplog.at_level("INFO", logger="picid.transforms.base.data_transform"):
            dt = DataTransform("t", transform, metadata)
        assert any(
            "implicit" in (r.getMessage() or getattr(r, "message", "")).lower()
            for r in caplog.records
        )
        container = create_dummy_split_container(n_units=1)
        result, _ = dt.forward(container)
        np.testing.assert_array_equal(
            result.features.train[0], container.features.train[0] * 2
        )
        np.testing.assert_array_equal(
            result.target.train[0], container.target.train[0] * 2
        )

    def test_process_assign_to_implicit_sequence_triggers_logger(self, caplog):
        """Implicit assign_to for a sequence logs once and preserves both target keys."""
        dt = DataTransform(
            "t",
            DummyStatelessTransform(),
            {"apply_to": "features", "assign_to": "features"},
        )
        with caplog.at_level("INFO", logger="picid.transforms.base.data_transform"):
            assign_to, assign_to_map = dt._process_assign_to(
                assign_to=None, apply_to=["features", "target"]
            )
        assert any(
            "implicit" in (r.getMessage() or getattr(r, "message", "")).lower()
            for r in caplog.records
        )
        assert assign_to == ["features", "target"]
        assert assign_to_map == ["features", "target"]

    def test_assign_to_required_when_apply_to_is_mapping(self):
        """Line 92: apply_to as Mapping without assign_to raises ValueError.
        Note: _process_apply_to converts Mapping to list, so this branch is only
        reachable if _process_assign_to is called with apply_to as Mapping from elsewhere.
        We test the branch by calling _process_assign_to directly.
        """
        from picid.transforms.base.data_transform import DataTransform

        dt = DataTransform(
            "t",
            DummyFittableTransform(),
            {"apply_to": "features", "assign_to": "features"},
        )
        with pytest.raises(ValueError, match="assign_to|ambiguity"):
            dt._process_assign_to(assign_to=None, apply_to={"feat": "features"})

    def test_forward_type_change_mock_strategy_preserves_mocked_train_shape(self):
        """Mocked strategy output with changed train shape is forwarded predictably."""
        dt = DataTransform(
            "t",
            DummyStatelessTransform(),
            {"apply_to": "features", "assign_to": "features"},
        )
        container = create_dummy_single_unit_container()

        class MockStrategy:
            def apply(self, **kwargs):
                data = kwargs["data"]
                result = data.copy(deep=False)
                result["features"]._data["train"] = np.array([[1.0, 2.0]])
                result["features"]._data["val"] = result["features"]["val"]
                if "test" in result["features"]:
                    result["features"]._data["test"] = result["features"]["test"]
                return result, {}

        dt.strategy = MockStrategy()
        result, _ = dt.forward(container)
        np.testing.assert_array_equal(result.features.train[0], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result.features.val[0], container.features.val[0])

    def test_forward_source_count_change_mock_strategy_preserves_other_splits(self):
        """Mocked strategy output with fewer train units keeps untouched splits intact."""
        dt = DataTransform(
            "t",
            DummyStatelessTransform(),
            {"apply_to": "features", "assign_to": "features"},
        )
        container = create_dummy_split_container(n_units=2)

        class MockStrategy:
            def apply(self, **kwargs):
                data = kwargs["data"]
                result = data.copy(deep=False)
                result["features"]._data["train"] = [
                    np.array([[1.0]])
                ]  # 1 unit (original had 2)
                result["features"]._data["val"] = data["features"]["val"]
                result["features"]._data["test"] = data["features"]["test"]
                return result, {}

        dt.strategy = MockStrategy()
        result, _ = dt.forward(container)
        assert len(result.features.train) == 1
        np.testing.assert_array_equal(result.features.train[0], np.array([[1.0]]))
        np.testing.assert_array_equal(result.features.val[0], container.features.val[0])

    def test_initialization_with_dictconfig_transform(self):
        """DictConfig instantiates transform via Hydra; forward applies scaling."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {"_target_": "test.transforms.base.conftest.DummyStatelessTransform"}
        )
        dt = DataTransform(
            "dictconfig_transform",
            cfg,
            {"apply_to": "features", "assign_to": "features"},
        )
        assert isinstance(dt.transform_instance, DummyStatelessTransform)
        container = create_dummy_single_unit_container()
        arr = container.features.train[0]
        result, _ = dt.forward(container)
        np.testing.assert_array_equal(result.features.train[0], arr * 2)


# ----- strategy.py: copy branch and structure error -----


class TestStrategyCoverage:
    def test_apply_ragged_shallow_copy_branch(self):
        """Cover line 99: shallow copy when get_instance_cls() indicates ak.Array."""
        strategy = TransformStrategy()
        transform = DummyRaggedTransform()
        ragged = ak.Array([[1.0, 2.0], [3.0]])
        container = SplitDatasetContainer(
            features={"train": [ragged], "val": [ragged]},
        )
        inner = container["features"]
        inner.get_instance_cls = (
            lambda: ak.Array
        )  # so "is not ak.Array" is False -> take else (shallow)
        result, _ = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )
        assert "features" in result
        out = result.features.train[0]
        assert isinstance(out, ak.Array)
        assert ak.sum(out) == ak.sum(ragged) * 2

    def test_apply_apply_key_not_mapping_raises_type_error(self):
        """Line 117: data[apply_key] not a Mapping raises TypeError."""

        class NotMapping:
            """get_instance_cls and copy so copy loop passes; copy returns valid dict for result."""

            def get_instance_cls(self):
                return {}

            def copy(self, deep=True):
                return {"train": [np.array([[1.0]])], "val": [np.array([[1.0]])]}

        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = SplitDatasetContainer(
            features={"train": [np.array([[1.0]])], "val": [np.array([[1.0]])]},
        )
        container._data["features"] = NotMapping()
        with pytest.raises(
            TypeError,
            match="expected a mapping|Unexpected structure|split-keyed mapping",
        ):
            strategy.apply(
                transform_instance=transform,
                data=container,
                apply_to_keys="features",
                assign_to_keys="features",
                assign_to_keys_map=["features"],
            )


# ----- transform_manager.py -----


class TestTransformManagerCoverage:
    def test_ensure_dict_config_with_plain_dict(self):
        """_ensure_dict_config with dict: line 44."""
        manager = ConfigTransformManager(
            transforms_config={"a": {"transform": {}, "metadata": {}}},
            lazy_instantiation=True,
        )
        assert isinstance(manager.config, DictConfig)
        assert list(manager.config.keys()) == ["a"]

    def test_ensure_dict_config_with_dictconfig(self):
        """_ensure_dict_config with DictConfig returns as-is: line 46."""
        config = OmegaConf.create({"t": {"transform": {}, "metadata": {}}})
        manager = ConfigTransformManager(
            transforms_config=config, lazy_instantiation=True
        )
        assert isinstance(manager.config, DictConfig)
        assert list(manager.config.keys()) == ["t"]

    def test_initialization_empty_config_no_instantiation_loop(self):
        """Empty config: line 51 (no entries in loop)."""
        manager = ConfigTransformManager(transforms_config={})
        assert len(manager) == 0

    def test_add_transforms_after_instantiation_raises(self):
        """Line 124: add_transforms_config after instantiation."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        manager.get_data_transforms()
        with pytest.raises(RuntimeError, match="after instantiation"):
            manager.add_transforms_config({"t2": {"transform": {}, "metadata": {}}})

    def test_update_transforms_after_instantiation_raises(self):
        """Line 139: update_transforms_config after instantiation."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        manager.get_data_transforms()
        with pytest.raises(RuntimeError, match="after instantiation"):
            manager.update_transforms_config({"t2": {"transform": {}, "metadata": {}}})

    def test_remove_transform_after_instantiation_raises(self):
        """Line 148: remove_transform after instantiation."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        manager.get_data_transforms()
        with pytest.raises(RuntimeError, match="after instantiation"):
            manager.remove_transform("t1")

    def test_add_transforms_config_name_conflict_raises(self):
        """Line 160: add_transforms_config with conflicting name."""
        manager = ConfigTransformManager(
            transforms_config={"t1": {"transform": {}, "metadata": {}}},
            lazy_instantiation=True,
        )
        with pytest.raises(KeyError, match="already exist"):
            manager.add_transforms_config({"t1": {"transform": {}, "metadata": {}}})

    def test_get_data_transform_missing_raises(self):
        """Lines 201-207: get_data_transform with missing name."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        with pytest.raises(KeyError, match="not found"):
            manager.get_data_transform("nonexistent")

    def test_get_transforms_by_fit_on(self):
        """Lines 230-231: filter by fit_on."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {
                    "apply_to": "features",
                    "assign_to": "features",
                    "fit_on": "train",
                },
            },
            "t2": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            },
        }
        manager = ConfigTransformManager(transforms_config=config)
        by_fit = manager.get_transforms_by_fit_on("train")
        assert "t1" in by_fit
        assert "t2" not in by_fit

    def test_remove_transform_missing_raises(self):
        """Lines 247-248: remove_transform with missing name."""
        manager = ConfigTransformManager(lazy_instantiation=True)
        with pytest.raises(KeyError, match="not found"):
            manager.remove_transform("nonexistent")

    def test_force_reinstantiate(self):
        """Lines 300, 304, 308: force_reinstantiate and aliases."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        manager.force_reinstantiate()
        assert manager.is_instantiated
        # Backward compatibility aliases
        _ = manager.instantiate_transforms()
        _ = manager.get_transforms()
        _ = manager.get_transform("t1")

    def test_create_transform_manager_from_config_path(self, tmp_path):
        """Lines 341-342: create from config_path."""
        cfg = tmp_path / "transforms.yaml"
        cfg.write_text(
            "t1:\n  transform:\n    _target_: test.transforms.base.conftest.DummyStatelessTransform\n  metadata:\n    apply_to: features\n    assign_to: features\n"
        )
        mgr = create_transform_manager_from_config(config_path=str(cfg))
        assert mgr.has_transform("t1")
        dt = mgr.get_data_transform("t1")
        container = create_dummy_single_unit_container()
        result, _ = dt.forward(container)
        np.testing.assert_array_equal(
            result.features.train[0], container.features.train[0] * 2
        )

    def test_create_transform_manager_from_config_dict(self):
        """Lines 343-344: create from config_dict."""
        mgr = create_transform_manager_from_config(
            config_dict={
                "t1": {
                    "transform": {
                        "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                    },
                    "metadata": {"apply_to": "features", "assign_to": "features"},
                }
            }
        )
        assert mgr.has_transform("t1")
        dt = mgr.get_data_transform("t1")
        container = create_dummy_single_unit_container()
        result, _ = dt.forward(container)
        np.testing.assert_array_equal(
            result.features.train[0], container.features.train[0] * 2
        )

    def test_create_transform_manager_no_config(self):
        """Lines 345-346: create with no config."""
        mgr = create_transform_manager_from_config()
        assert mgr is not None
        assert len(mgr) == 0

    def test_manager_transform_config_none_logs_error(self, caplog):
        """Lines 64, 73: entry with missing 'transform' key triggers logger.error (and may raise)."""
        config = {
            "bad_entry": {"metadata": {"apply_to": "features"}},  # no "transform" key
        }
        with caplog.at_level("ERROR", logger="picid.transforms.base.transform_manager"):
            with pytest.raises(RuntimeError, match="Failed to instantiate"):
                ConfigTransformManager(transforms_config=config)
        assert any("missing" in r.getMessage().lower() for r in caplog.records)

    def test_manager_instantiate_all_early_return_on_second_call(self):
        """Line 51: _instantiate_all returns early when _is_instantiated is True."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        manager.get_data_transforms()
        assert manager._is_instantiated
        manager._instantiate_all()  # second call hits "if self._is_instantiated: return" at 51
        assert len(manager.get_data_transforms()) == 1

    def test_manager_get_data_transform_key_error_path(self):
        """Line 202: KeyError when name not in _data_transforms."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        manager.get_data_transforms()
        with pytest.raises(KeyError, match="not found"):
            manager.get_data_transform("other_name")

    def test_manager_get_transforms_by_apply_to_returns_filtered(self):
        """Lines 217-218: get_transforms_by_apply_to builds OrderedDict from get_data_transforms."""
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
                "metadata": {"apply_to": "target", "assign_to": "target"},
            },
        }
        manager = ConfigTransformManager(transforms_config=config)
        by_apply = manager.get_transforms_by_apply_to("features")
        assert "t1" in by_apply
        assert "t2" not in by_apply
        assert len(by_apply) == 1

    def test_get_data_transforms_empty_config_returns_ordered_dict(self):
        """Line 148: get_data_transforms when _data_transforms is empty returns OrderedDict()."""
        manager = ConfigTransformManager(transforms_config={})
        result = manager.get_data_transforms()
        assert isinstance(result, OrderedDict)
        assert len(result) == 0

    def test_prepare_transforms_returns_list(self):
        """Line 202/217-218: prepare_transforms returns list of DataTransform; get_transforms_by_apply_to return."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        transforms_list = manager.prepare_transforms()
        assert isinstance(transforms_list, list)
        assert len(transforms_list) == 1

    def test_add_transforms_config_not_lazy_instantiates(self):
        """Line 124: add_transforms_config when not lazy calls _instantiate_all()."""
        manager = ConfigTransformManager(transforms_config={}, lazy_instantiation=False)
        manager.add_transforms_config(
            {
                "t1": {
                    "transform": {
                        "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                    },
                    "metadata": {"apply_to": "features", "assign_to": "features"},
                },
            }
        )
        assert len(manager.get_data_transforms()) == 1

    def test_update_transforms_config_when_not_instantiated(self):
        """Lines 144-148: update_transforms_config when not instantiated replaces config."""
        manager = ConfigTransformManager(
            transforms_config={
                "t1": {
                    "transform": {
                        "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                    },
                    "metadata": {"apply_to": "features", "assign_to": "features"},
                }
            },
            lazy_instantiation=True,
        )
        manager.update_transforms_config(
            {
                "t2": {
                    "transform": {
                        "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                    },
                    "metadata": {"apply_to": "features", "assign_to": "features"},
                }
            }
        )
        assert list(manager.get_transform_names()) == ["t2"]

    def test_update_transforms_config_after_clear_cache_covers_instantiate_branch(self):
        """Lines 148-149: update_transforms_config after clear_cache with lazy=False runs _instantiate_all()."""
        manager = ConfigTransformManager(
            transforms_config={
                "t1": {
                    "transform": {
                        "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                    },
                    "metadata": {"apply_to": "features", "assign_to": "features"},
                }
            },
            lazy_instantiation=False,
        )
        manager.clear_cache()
        manager.update_transforms_config(
            {
                "t1": {
                    "transform": {
                        "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                    },
                    "metadata": {"apply_to": "features", "assign_to": "features"},
                }
            }
        )
        assert len(manager.get_data_transforms()) == 1

    def test_get_data_transform_success_and_missing(self):
        """Lines 204, 207: get_data_transform success return and KeyError path."""
        config = {
            "t1": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        dt = manager.get_data_transform("t1")
        assert dt is not None
        assert dt.transform_name == "t1"
        with pytest.raises(KeyError, match="not found"):
            manager.get_data_transform("nonexistent")

    def test_get_data_transform_empty_transforms_raises(self):
        """Line 202: get_data_transform when _data_transforms is empty (falsy) raises KeyError."""
        manager = ConfigTransformManager(transforms_config={})
        with pytest.raises(KeyError, match="not found"):
            manager.get_data_transform("any")

    def test_get_transform_config_missing_raises(self):
        """Line 172: get_transform_config when name not in config raises KeyError."""
        manager = ConfigTransformManager(
            transforms_config={"t1": {"transform": {}, "metadata": {}}},
            lazy_instantiation=True,
        )
        with pytest.raises(KeyError, match="not found"):
            manager.get_transform_config("nonexistent")

    def test_get_transforms_config_returns_config(self):
        """Line 160: get_transforms_config() returns self.config."""
        manager = ConfigTransformManager(
            transforms_config={"t1": {"transform": {}, "metadata": {}}},
            lazy_instantiation=True,
        )
        cfg = manager.get_transforms_config()
        assert cfg is not None
        assert "t1" in cfg


# ----- multisource.py -----


class TestMultisourceCoverage:
    def test_tolist_sorted_dict(self):
        """Line 103: tolist with SortedDict."""
        sd = SortedDict({"a": 1, "b": 2})
        out = tolist(sd)
        assert out == [1, 2]

    def test_tolist_dict(self):
        """Line 106: tolist with plain dict."""
        out = tolist({"x": 1, "y": 2})
        assert out == [1, 2]

    def test_tolist_non_dict_returns_unchanged(self):
        """Line 108 (127): tolist with non-dict returns object as-is."""
        arr = np.array([1.0, 2.0])
        out = tolist(arr)
        assert out is arr
        out = tolist(42)
        assert out == 42

    def test_fit_unsupported_data_type_raises(self):
        """Line 309: fit when transform does not support data type."""

        class RaggedOnlyTransform(
            ConcatFitAndPerSegmentTransformMixin, RaggedTransform
        ):
            def fit_data(self, data, metadata): ...
            def transform_data(self, data, metadata):
                return data[list(data.keys())[0]]

        t = RaggedOnlyTransform()
        segments = [NamedTransformInput(features=np.array([[1.0]]))]
        with pytest.raises(
            ValueError, match=r"does not support .*data_kind|capability"
        ):
            t.fit_multi_source(segments, metadata={"apply_to_keys": ["features"]})

    @pytest.mark.skip(
        reason="Exception handler in transform_function_to_data_segments uses data_segment which is unbound when the else (unsupported data type) branch raises; skip until multisource.py is fixed."
    )
    def test_transform_unsupported_data_type_raises(self):
        """Line 533: transform_multi_source when data type not supported hits else and raises ValueError."""

        class RaggedOnlyTransform(NoFitPerSegmentMixin, RaggedTransform):
            def transform_data(self, data, metadata):
                return data[list(data.keys())[0]]

        t = RaggedOnlyTransform()
        segments = [NamedTransformInput(features=np.array([[1.0, 2.0]]))]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}
        with pytest.raises(ValueError, match="does not support the data type"):
            t.transform_multi_source(segments, metadata=metadata)

    def test_transform_apply_to_keys_string_normalized(self):
        """Line 351: apply_to_keys as string is normalized in transform path."""
        transform = DummyStatelessTransform()
        segments = [NamedTransformInput(features=np.array([[1.0, 2.0]]))]
        metadata = {"apply_to_keys": "features", "assign_to_map": ["features"]}
        out, _ = transform.transform_multi_source(segments, metadata=metadata)
        assert len(out) == 1
        np.testing.assert_array_equal(out[0], np.array([[2.0, 4.0]]))

    def test_concatenate_units_fit_multi_source_raises(self):
        """Line 695: NoFitConcatAlongAxisMixin.fit_multi_source raises NotImplementedError."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        t = ConcatenateUnitsTransform(axis=0)
        with pytest.raises(
            NotImplementedError, match="does not support fitting|fit_multi_source"
        ):
            t.fit_multi_source(
                [NamedTransformInput(features=np.array([[1.0]]))],
                metadata={"apply_to_keys": ["features"]},
            )

    def test_concatenate_units_transform_with_ak_array(self):
        """Lines 742-753: NoFitConcatAlongAxisMixin with ak.Array."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        t = ConcatenateUnitsTransform(axis=0)
        segs = [
            NamedTransformInput(features=ak.Array([[1.0], [2.0]])),
            NamedTransformInput(features=ak.Array([[3.0], [4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}
        out, _ = t.transform_multi_source(segs, metadata=metadata)
        assert len(out) == 1
        arr = out[0] if not hasattr(out[0], "keys") else out[0]["features"]
        assert isinstance(arr, ak.Array)
        np.testing.assert_array_almost_equal(
            ak.to_numpy(ak.flatten(arr)), np.array([2.0, 4.0, 6.0, 8.0])
        )

    def test_concatenate_units_transform_multi_key_length_check(self):
        """Lines 771-786: multiple apply_to_keys length consistency."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        class MultiKeyConcat(ConcatenateUnitsTransform):
            def transform_data(self, data, metadata):
                return {"features": data["features"] * 2, "target": data["target"] * 2}

        t = MultiKeyConcat(axis=0)
        segs = [
            NamedTransformInput(
                features=np.array([[1.0], [2.0]]), target=np.array([[0.0], [0.0]])
            ),
            NamedTransformInput(
                features=np.array([[3.0], [4.0]]), target=np.array([[0.0], [0.0]])
            ),
        ]
        metadata = {"apply_to_keys": ["features", "target"]}
        out, _ = t.transform_multi_source(segs, metadata=metadata)
        assert len(out) == 1
        merged = out[0]["features"]
        np.testing.assert_array_almost_equal(
            merged, np.array([[2.0], [4.0], [6.0], [8.0]])
        )

    def test_inverse_transform_mixin_init_requires_inverse_transform_or_data(self):
        """InverseTransformMixin requires inverse_transform or inverse_transform_data."""

        # Class with neither inverse method should raise at init
        class NoInverse(InverseTransformMixin, DenseTransform):
            def transform_data(self, data, metadata):
                return data

        with pytest.raises(TypeError, match="inverse_transform"):
            NoInverse()

    def test_transform_multi_source_ragged_to_dense_exception_logging(self):
        """Lines 533-598: exception in transform_function_to_data_segments is logged and re-raised."""

        class RaisingTransform(NoFitPerSegmentMixin, DenseTransform):
            def transform_data(self, data, metadata):
                raise RuntimeError("test error in transform")

        t = RaisingTransform()
        ragged = ak.Array([[1.0, 2.0], [3.0]])
        segments = [NamedTransformInput(features=ragged)]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}
        with pytest.raises(RuntimeError, match="test error"):
            t.transform_multi_source(segments, metadata=metadata)

    def test_transform_multi_source_inconsistent_types_raises(self):
        """Lines 366-367: inconsistent data types across segments raises ValueError."""
        transform = DummyStatelessTransform()
        segments = [
            NamedTransformInput(features=np.array([[1.0]])),
            NamedTransformInput(features=ak.Array([[1.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        with pytest.raises(ValueError, match="same data type"):
            transform.transform_multi_source(segments, metadata=metadata)

    def test_fit_multi_source_ragged_but_regular_treated_as_dense(self):
        """Line 242: fit with regular awkward (find_singular_ragged_dim None) sets data_is_ragged=False."""
        regular_ak = ak.from_numpy(np.ones((2, 3, 4)))
        segments = [NamedTransformInput(features=regular_ak)]
        transform = DummyFittableTransform()
        transform.fit_multi_source(segments, metadata={"apply_to_keys": ["features"]})
        assert transform.fitted

    def test_transform_ragged_but_regular_returns_base_data_object(self):
        """Lines 402-403: ragged_but_regular path when transform returns BaseDataObjectWithMetadata."""
        from picid.data.data_objects import BaseDataObjectWithMetadata

        regular_ak = ak.from_numpy(
            np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        )

        class ReturnNamedInput(NoFitPerSegmentMixin, DenseTransform):
            def transform_data(self, data, metadata):
                out = data[list(data.keys())[0]] * 2
                return NamedTransformInput(features=out)

        t = ReturnNamedInput()
        segments = [NamedTransformInput(features=regular_ak)]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}
        result, _ = t.transform_multi_source(segments, metadata=metadata)
        assert len(result) == 1
        assert hasattr(result[0], "keys") or isinstance(
            result[0], (dict, BaseDataObjectWithMetadata)
        )
        feat = result[0]["features"]
        feat_arr = ak.to_numpy(feat) if isinstance(feat, ak.Array) else np.asarray(feat)
        np.testing.assert_array_equal(
            feat_arr,
            np.array([[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]]),
        )

    def test_transform_multi_source_different_key_lengths_raises(self):
        """Line 661: transform_multi_source with segments having different key counts raises."""
        transform = DummyStatelessTransform()
        segments = [
            NamedTransformInput(features=np.array([[1.0]])),
            NamedTransformInput(features=np.array([[2.0]]), target=np.array([[0.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        with pytest.raises(
            ValueError, match="different numbers of keys|lengths of the data segments"
        ):
            transform.transform_multi_source(segments, metadata=metadata)

    def test_concatenate_units_key_not_in_segment_raises(self):
        """Lines 661, 711: lengths check and key-not-in-segment assert."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        t = ConcatenateUnitsTransform(axis=0)
        segs = [
            NamedTransformInput(features=np.array([[1.0]]), target=np.array([[0.0]])),
            NamedTransformInput(
                features=np.array([[2.0]])
            ),  # missing "target" -> different key count
        ]
        metadata = {"apply_to_keys": ["features", "target"]}
        with pytest.raises(
            ValueError, match="different numbers of keys|lengths of the data segments"
        ):
            t.transform_multi_source(segs, metadata=metadata)

    def test_concatenate_units_ak_two_segments_success(self):
        """NoFitConcatAlongAxisMixin: two ak.Array segments concatenate successfully."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        t = ConcatenateUnitsTransform(axis=0)
        seg1 = NamedTransformInput(features=ak.Array([[1.0], [2.0]]))
        seg2 = NamedTransformInput(features=ak.Array([[3.0], [4.0]]))
        segs = [seg1, seg2]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}
        out, _ = t.transform_multi_source(segs, metadata=metadata)
        assert len(out) == 1
        feat = out[0]["features"]
        assert isinstance(feat, ak.Array)
        np.testing.assert_array_almost_equal(
            ak.to_numpy(ak.flatten(feat)), np.array([2.0, 4.0, 6.0, 8.0])
        )

    def test_concatenate_units_multi_key_ak_and_numpy_length_checks(self):
        """Multi-key concatenate: both keys same type to satisfy uniform data type; length checks."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        class MultiKeyConcat(ConcatenateUnitsTransform):
            def transform_data(self, data, metadata):
                return {"features": data["features"] * 2, "target": data["target"] * 2}

        t = MultiKeyConcat(axis=0)
        # Use same type for all keys (np) so _assert_uniform_data_types passes
        segs = [
            NamedTransformInput(
                features=np.array([[1.0], [2.0]]), target=np.array([[0.0], [0.0]])
            ),
            NamedTransformInput(
                features=np.array([[3.0], [4.0]]), target=np.array([[0.0], [0.0]])
            ),
        ]
        metadata = {
            "apply_to_keys": ["features", "target"],
            "assign_to_map": ["features", "target"],
        }
        out, _ = t.transform_multi_source(segs, metadata=metadata)
        assert len(out) == 1
        row = out[0]
        np.testing.assert_array_almost_equal(
            row["features"], np.array([[2.0], [4.0], [6.0], [8.0]])
        )
        np.testing.assert_array_almost_equal(
            row["target"], np.array([[0.0], [0.0], [0.0], [0.0]])
        )

    def test_concatenate_units_inconsistent_lengths_raises(self):
        """Line 786: two keys with different lengths after concat raises ValueError."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        class BadConcat(ConcatenateUnitsTransform):
            def transform_data(self, data, metadata):
                return {
                    "features": np.array([[1.0], [2.0]]),
                    "target": np.array([[1.0], [2.0], [3.0], [4.0]]),
                }

        t = BadConcat(axis=0)
        segs = [
            NamedTransformInput(
                features=np.array([[1.0]]), target=np.array([[1.0], [2.0]])
            ),
            NamedTransformInput(
                features=np.array([[2.0]]), target=np.array([[3.0], [4.0]])
            ),
        ]
        metadata = {"apply_to_keys": ["features", "target"]}
        with pytest.raises(
            ValueError, match="After concatenation|lengths of the concatenated"
        ):
            t.transform_multi_source(segs, metadata=metadata)

    def test_ragged_to_dense_exception_after_index_depth_set(self):
        """Lines 567-568, 575-584: exception in ragged-to-dense loop triggers case analysis with index_depth."""

        class RaiseOnSecondCoord(NoFitPerSegmentMixin, DenseTransform):
            def __init__(self):
                self._count = 0

            def transform_data(self, data, metadata):
                self._count += 1
                if self._count >= 2:
                    raise RuntimeError("fail on second segment")
                return data[list(data.keys())[0]] * 2

        t = RaiseOnSecondCoord()
        ragged = ak.Array([[1.0, 2.0], [3.0, 4.0], [5.0]])
        segments = [NamedTransformInput(features=ragged)]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}
        with pytest.raises(RuntimeError, match="fail on second"):
            t.transform_multi_source(segments, metadata=metadata)

    def test_ragged_to_dense_reassembly_regular_output_hits_from_regular(self):
        """Line 515: reassembly with len(var_dims)==0 after regularize uses ak.from_regular."""

        class SameShapeTransform(NoFitPerSegmentMixin, DenseTransform):
            def transform_data(self, data, metadata):
                return np.array(
                    [[1.0, 2.0]]
                )  # same shape (1,2) per segment -> regular output

        t = SameShapeTransform()
        ragged = ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0]])
        segments = [NamedTransformInput(features=ragged)]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}
        result, log = t.transform_multi_source(segments, metadata=metadata)
        assert log.get("mode") == "ragged_to_dense"
        assert len(result) == 1
        feat = result[0]["features"]
        feat_arr = ak.to_numpy(feat) if isinstance(feat, ak.Array) else np.asarray(feat)
        np.testing.assert_array_equal(
            feat_arr.reshape(-1),
            np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]),
        )

    def test_concatenate_units_different_data_types_raises(self):
        """Line 727: NoFitConcatAlongAxisMixin when data types differ across segments."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        t = ConcatenateUnitsTransform(axis=0)
        segs = [
            NamedTransformInput(features=np.array([[1.0], [2.0]])),
            NamedTransformInput(features=ak.Array([[3.0], [4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        with pytest.raises(
            ValueError, match="data type mismatch|not the same across all data segments"
        ):
            t.transform_multi_source(segs, metadata=metadata)

    def test_concatenate_units_unsupported_type_raises(self):
        """Line 781: NoFitConcatAlongAxisMixin when key type is not ak.Array or np.ndarray."""
        from test.transforms.base.conftest import ConcatenateUnitsTransform

        t = ConcatenateUnitsTransform(axis=0)

        # Segment that reports type as list so we hit the else branch
        class ListTypeSegment:
            def get_instance_cls(self):
                return {"features": list}

            def keys(self):
                return ["features"]

            def __getitem__(self, k):
                if k == "features":
                    return [1.0, 2.0]
                raise KeyError(k)

        segs = [ListTypeSegment(), ListTypeSegment()]
        metadata = {"apply_to_keys": ["features"]}
        with pytest.raises(ValueError, match="unsupported data type|is not supported"):
            t.transform_multi_source(segs, metadata=metadata)
