"""Tests for Phase 4.3: integrity guards (IntegrityPolicy, run_check, IntegrityCheckStep)."""

import pytest

from picid.transforms.base.integrity import run_check, run_checks
from picid.transforms.base.pipeline import TransformContext, IntegrityCheckStep

from test.transforms.base.conftest import (
    create_dummy_single_unit_container,
    DummyStatelessTransform,
)


class TestRunCheck:
    def test_fail_policy_raises(self):
        def bad():
            raise ValueError("expected")

        with pytest.raises(ValueError, match="expected"):
            run_check("test", bad, "fail")

    def test_warn_policy_warns(self):
        def bad():
            raise ValueError("expected")

        with pytest.warns(UserWarning, match="Integrity check 'test' failed"):
            run_check("test", bad, "warn")

    def test_allow_policy_suppresses(self):
        def bad():
            raise ValueError("expected")

        run_check("test", bad, "allow")  # no raise, no warn

    def test_pass_no_raise(self):
        run_check("ok", lambda: None, "fail")
        run_check("ok", lambda: None, "warn")
        run_check("ok", lambda: None, "allow")


class TestRunChecks:
    def test_runs_in_order_stops_on_fail(self):
        seen = []

        def a():
            seen.append("a")

        def b():
            raise ValueError("b")

        def c():
            seen.append("c")

        with pytest.raises(ValueError, match="b"):
            run_checks(
                [
                    ("a", a, "allow"),
                    ("b", b, "fail"),
                    ("c", c, "allow"),
                ]
            )
        assert seen == ["a"]
        assert "c" not in seen

    def test_warn_continues(self):
        seen = []

        def a():
            seen.append("a")

        def b():
            raise ValueError("b")

        def c():
            seen.append("c")

        with pytest.warns(UserWarning):
            run_checks(
                [
                    ("a", a, "allow"),
                    ("b", b, "warn"),
                    ("c", c, "allow"),
                ]
            )
        assert seen == ["a", "c"]


class TestIntegrityCheckStep:
    def test_no_checks_no_op(self):
        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
        )
        IntegrityCheckStep().run(ctx)
        assert ctx.transformed_data is None  # unchanged

    def test_with_checks_fail_runs_and_raises(self):
        def raise_intended():
            raise ValueError("intended")

        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
            integrity_checks=[
                ("must_fail", raise_intended, "fail"),
            ],
        )
        with pytest.raises(ValueError, match="intended"):
            IntegrityCheckStep().run(ctx)

    def test_with_checks_allow_no_raise(self):
        ran = []

        def mark():
            ran.append(1)

        def raise_ignored():
            raise ValueError("ignored")

        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
            integrity_checks=[
                ("one", mark, "allow"),
                ("two", raise_ignored, "allow"),
            ],
        )
        IntegrityCheckStep().run(ctx)
        assert ran == [1]
