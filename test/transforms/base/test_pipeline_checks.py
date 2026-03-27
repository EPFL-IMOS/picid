"""Tests for picid.transforms.base.pipeline.checks."""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.pipeline.checks import (
    _run_scoped_checks,
    _default_before_check_nans,
    _default_before_check_lengths,
    _default_after_check_structure,
    _default_after_check_splits,
    build_default_before_checks,
    build_default_after_checks,
)


class TestRunScopedChecks:
    def test_no_arg_check_deprecation_warning(self):
        """No-arg check triggers DeprecationWarning (lines 51-58)."""

        def no_arg_fn():
            pass

        checks = [("noarg", no_arg_fn, "fail")]
        with pytest.warns(DeprecationWarning, match="no-arg signature"):
            _run_scoped_checks(checks, {"x": 1})
        assert "noarg" in _run_scoped_checks(checks, {"x": 1})

    def test_with_arg_check_runs(self):
        """Check with 1-arg runs and gets data_slice (line 60)."""
        seen = []

        def one_arg_fn(data):
            seen.append(data)

        checks = [("with_arg", one_arg_fn, "fail")]
        data = {"split": "train"}
        ran = _run_scoped_checks(checks, data)
        assert ran == ["with_arg"]
        assert seen == [data]

    def test_policy_fail_raises_runtime_error(self):
        """policy 'fail' raises RuntimeError with message (lines 63-77)."""

        def failing_fn(_):
            raise ValueError("check failed")

        checks = [("fail_check", failing_fn, "fail")]
        with pytest.raises(RuntimeError, match="Integrity check.*failed"):
            _run_scoped_checks(checks, None)

    def test_policy_fail_includes_step_id_in_message(self):
        """RuntimeError includes step_id when provided (lines 67-68)."""

        def failing_fn(_):
            raise ValueError("err")

        checks = [("c", failing_fn, "fail")]
        with pytest.raises(RuntimeError, match="transform.*foo"):
            _run_scoped_checks(checks, None, step_id="foo")

    def test_policy_fail_includes_split_in_message(self):
        """RuntimeError includes split when provided (line 70)."""

        def failing_fn(_):
            raise ValueError("err")

        checks = [("c", failing_fn, "fail")]
        with pytest.raises(RuntimeError, match="split.*train"):
            _run_scoped_checks(checks, None, split="train")

    def test_policy_fail_includes_stage_in_message(self):
        """RuntimeError includes stage when provided (line 66)."""

        def failing_fn(_):
            raise ValueError("err")

        checks = [("c", failing_fn, "fail")]
        with pytest.raises(RuntimeError, match="before-transform"):
            _run_scoped_checks(checks, None, stage="before-transform")

    def test_policy_warn_logs_and_continues(self, caplog):
        """policy 'warn' logs and appends name to ran, does not raise (lines 78-82)."""

        def failing_fn(_):
            raise ValueError("warn check")

        checks = [("warn_check", failing_fn, "warn")]
        ran = _run_scoped_checks(checks, None)
        assert ran == ["warn_check"]
        assert "warn_check" in caplog.text or "warn" in caplog.text.lower()


class TestDefaultBeforeChecks:
    def test_before_check_nans_raises_on_nan(self):
        """_default_before_check_nans raises when NaN present (lines 91-96)."""
        chunk = NamedTransformInput(features=np.array([[1.0, np.nan]]))
        split_chunks = {"train": [chunk]}
        with pytest.raises(ValueError, match="NaN"):
            _default_before_check_nans(split_chunks)

    def test_before_check_nans_passes_clean(self):
        """_default_before_check_nans passes with no NaN."""
        chunk = NamedTransformInput(features=np.array([[1.0, 2.0]]))
        split_chunks = {"train": [chunk]}
        _default_before_check_nans(split_chunks)

    def test_before_check_lengths_raises_on_mismatch(self):
        """_default_before_check_lengths raises on length mismatch (lines 104-107)."""
        chunk = NamedTransformInput(
            a=np.zeros(10),
            b=np.zeros(5),
        )
        split_chunks = {"train": [chunk]}
        with pytest.raises(ValueError, match="Length|mismatch|consistent"):
            _default_before_check_lengths(split_chunks)

    def test_before_check_lengths_passes_consistent(self):
        """_default_before_check_lengths passes when lengths match."""
        chunk = NamedTransformInput(
            a=np.zeros(10),
            b=np.zeros(10),
        )
        split_chunks = {"train": [chunk]}
        _default_before_check_lengths(split_chunks)


class TestDefaultAfterChecks:
    def test_after_check_structure_raises_on_non_mapping(self):
        """_default_after_check_structure raises when value not Mapping (lines 115-120)."""
        assign_to_slice = {"features": "not a dict"}
        with pytest.raises(TypeError, match="split-keyed mapping"):
            _default_after_check_structure(assign_to_slice)

    def test_after_check_structure_passes(self):
        """_default_after_check_structure passes with Mapping values."""
        assign_to_slice = {"features": {"train": [1], "val": [1]}}
        _default_after_check_structure(assign_to_slice)

    def test_after_check_splits_raises_on_changed_splits(self):
        """_default_after_check_splits raises when split set changed (lines 131-136)."""
        assign_to_slice = {"features": {"train": [1]}}
        original_splits = frozenset({"train", "val"})
        with pytest.raises(KeyError, match="split set changed"):
            _default_after_check_splits(assign_to_slice, original_splits)

    def test_after_check_splits_passes_when_unchanged(self):
        """_default_after_check_splits passes when splits match."""
        assign_to_slice = {"features": {"train": [1], "val": [1]}}
        original_splits = frozenset({"train", "val"})
        _default_after_check_splits(assign_to_slice, original_splits)


class TestBuildDefaultChecks:
    def test_build_default_before_checks(self):
        """build_default_before_checks returns before checks (lines 146-149)."""
        checks = build_default_before_checks(policy="fail")
        assert len(checks) == 2
        assert checks[0][0] == "check_no_nans"
        assert checks[1][0] == "check_length_consistency"
        assert all(c[2] == "fail" for c in checks)

    def test_build_default_after_checks(self):
        """build_default_after_checks returns after checks (lines 164-170)."""
        original_splits = frozenset({"train", "val"})
        checks = build_default_after_checks(original_splits, policy="fail")
        assert len(checks) == 2
        assert checks[0][0] == "check_output_structure"
        assert checks[1][0] == "check_split_preservation"
