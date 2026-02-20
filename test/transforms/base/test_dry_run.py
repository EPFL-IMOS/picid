"""Tests for Phase 5.2: structural dry run (dry_run_transforms)."""
import pytest

from picid.transforms.base.dry_run import (
    DryRunResult,
    dry_run_transforms,
    _get_first_segment,
)
from picid.transforms.base.transform_manager import ConfigTransformManager
from picid.data.data_objects.data import SplitDatasetContainer

from test.transforms.base.conftest import (
    create_dummy_single_unit_container,
    create_dummy_split_container,
    DummyStatelessTransform,
)


class TestDryRunResult:
    def test_str_ok(self):
        r = DryRunResult(transform_name="t1", success=True, issues=[])
        assert "OK" in str(r)
        assert "t1" in str(r)

    def test_str_fail_with_issues(self):
        r = DryRunResult(transform_name="t1", success=False, issues=["key x not in data"])
        assert "FAIL" in str(r)
        assert "key x not in data" in str(r)


class TestGetFirstSegment:
    def test_returns_none_for_missing_key(self):
        container = create_dummy_single_unit_container()
        seg = _get_first_segment(container, ["nonexistent"])
        assert seg is None

    def test_returns_segment_when_data_present(self):
        container = create_dummy_single_unit_container()
        seg = _get_first_segment(container, ["features"])
        assert seg is not None
        assert "features" in seg


class TestDryRunTransforms:
    def test_empty_manager_returns_empty_list(self):
        manager = ConfigTransformManager(transforms_config={})
        container = create_dummy_single_unit_container()
        results = dry_run_transforms(manager, container)
        assert results == []

    def test_single_transform_matching_data_returns_success(self):
        config = {
            "scale": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        container = create_dummy_single_unit_container()
        results = dry_run_transforms(manager, container)
        assert len(results) == 1
        assert results[0].transform_name == "scale"
        assert results[0].success is True
        assert results[0].issues == []

    def test_transform_missing_apply_to_key_returns_failure(self):
        config = {
            "scale": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "target", "assign_to": "target"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        container = create_dummy_single_unit_container()  # has "features", not "target"
        results = dry_run_transforms(manager, container)
        assert len(results) == 1
        assert results[0].success is False
        assert any("target" in i for i in results[0].issues)

    def test_apply_to_key_produced_by_earlier_transform_succeeds_with_skipped_handler_check(
        self,
    ):
        """Keys created by an earlier transform (assign_to) are valid for later apply_to."""
        config = {
            "rename_target_to_rul": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "target", "assign_to": "rul"},
            },
            "scale_rul": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "rul", "assign_to": "rul"},
            },
        }
        manager = ConfigTransformManager(transforms_config=config)
        container = create_dummy_split_container()  # has "features", "target"
        results = dry_run_transforms(manager, container)
        assert len(results) == 2
        assert results[0].success is True
        assert results[0].issues == []
        assert results[1].success is True
        assert any(
            "produced by earlier transform" in i for i in results[1].issues
        ), "second transform should report handler check skipped (rul not in initial container)"
