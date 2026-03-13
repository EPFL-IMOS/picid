"""Tests for Phase 5.2: structural dry run (dry_run_transforms)."""

from unittest.mock import patch

from picid.transforms.base.dry_run import (
    DryRunResult,
    dry_run_transforms,
    _get_first_segment,
)
from picid.transforms.base.dry_run.dry_run import _normalize_keys
from picid.transforms.base.transform_manager import ConfigTransformManager

from test.transforms.base.conftest import (
    create_dummy_single_unit_container,
    create_dummy_split_container,
)


class TestDryRunResult:
    def test_str_ok(self):
        r = DryRunResult(transform_name="t1", success=True, issues=[])
        assert "OK" in str(r)
        assert "t1" in str(r)

    def test_str_fail_with_issues(self):
        r = DryRunResult(
            transform_name="t1", success=False, issues=["key x not in data"]
        )
        assert "FAIL" in str(r)
        assert "key x not in data" in str(r)


class TestNormalizeKeys:
    def test_normalize_keys_none_returns_empty(self):
        assert _normalize_keys(None) == []

    def test_normalize_keys_str_returns_list(self):
        assert _normalize_keys("features") == ["features"]

    def test_normalize_keys_list_passthrough(self):
        assert _normalize_keys(["a", "b"]) == ["a", "b"]


class TestGetFirstSegment:
    def test_returns_none_for_empty_apply_to(self):
        container = create_dummy_single_unit_container()
        assert _get_first_segment(container, []) is None

    def test_returns_none_for_missing_key(self):
        container = create_dummy_single_unit_container()
        seg = _get_first_segment(container, ["nonexistent"])
        assert seg is None

    def test_returns_none_when_split_data_not_mapping(self):
        container = create_dummy_single_unit_container()
        container._data["bad_key"] = "not a dict"
        seg = _get_first_segment(container, ["bad_key"])
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

    def test_transform_missing_apply_to_returns_failure(self):
        """Transform with no apply_to gets failure (lines 113-115)."""
        config = {
            "scale": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": "features", "assign_to": "features"},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        dt = list(manager.get_data_transforms().values())[0]
        saved = dt.apply_to
        dt.apply_to = None
        try:
            container = create_dummy_single_unit_container()
            results = dry_run_transforms(manager, container)
            assert len(results) == 1
            assert results[0].success is False
            assert "missing apply_to" in str(results[0].issues)
        finally:
            dt.apply_to = saved

    def test_transform_apply_to_list_implicit_assign_to(self):
        """When assign_to is empty but apply_to has keys, assign_to = apply_to (line 119)."""
        config = {
            "scale": {
                "transform": {
                    "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
                },
                "metadata": {"apply_to": ["features"]},
            }
        }
        manager = ConfigTransformManager(transforms_config=config)
        container = create_dummy_single_unit_container()
        results = dry_run_transforms(manager, container)
        assert len(results) == 1
        assert results[0].success is True

    def test_transform_data_not_split_mapping_returns_failure(self):
        """When container key is not a Mapping, add issue and fail (lines 144-148)."""
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
        container._data["features"] = "not a mapping"
        results = dry_run_transforms(manager, container)
        assert len(results) == 1
        assert results[0].success is False
        assert any("not a split mapping" in i for i in results[0].issues)

    def test_infer_data_kind_failure_returns_failure(self):
        """When infer_data_kind raises, append failure (lines 163-167)."""
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
        with patch(
            "picid.transforms.base.dry_run.dry_run.infer_data_kind",
            side_effect=ValueError("infer failed"),
        ):
            results = dry_run_transforms(manager, container)
        assert len(results) == 1
        assert results[0].success is False
        assert "infer_data_kind failed" in results[0].issues[0]

    def test_get_capability_failure_returns_failure(self):
        """When get_capability raises, append failure (lines 171-175)."""
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
        with patch(
            "picid.transforms.base.dry_run.dry_run.get_capability",
            side_effect=TypeError("capability failed"),
        ):
            results = dry_run_transforms(manager, container)
        assert len(results) == 1
        assert results[0].success is False
        assert "get_capability failed" in results[0].issues[0]

    def test_get_handler_keyerror_returns_failure(self):
        """When get_handler raises KeyError, append failure (lines 179-183)."""
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
        with patch(
            "picid.transforms.base.dry_run.dry_run.get_handler",
            side_effect=KeyError("no handler"),
        ):
            results = dry_run_transforms(manager, container)
        assert len(results) == 1
        assert results[0].success is False
        assert "no handler for" in results[0].issues[0]

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
