"""Tests for Phase 3: pipeline steps and lifecycle hooks."""

import pytest

from picid.transforms.base.pipeline import (
    TransformContext,
    CopyStep,
    register_hook,
    clear_hooks,
    _emit,
)

from test.transforms.base.conftest import (
    create_dummy_single_unit_container,
    DummyStatelessTransform,
    DummyFittableTransform,
)


class TestTransformContext:
    """TransformContext creation and defaults."""

    def test_context_creation(self):
        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
        )
        assert ctx.data is container
        assert ctx.transformed_data is None
        assert ctx.available_splits is None
        assert ctx.log == {}

    def test_context_with_fit_and_strategy(self):
        container = create_dummy_single_unit_container()
        strategy = object()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyFittableTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
            fit_on_split="train",
            fit_on_key="features",
            strategy=strategy,
        )
        assert ctx.fit_on_split == "train"
        assert ctx.fit_on_key == "features"
        assert ctx.strategy is strategy


class TestCopyStep:
    """CopyStep run with valid context."""

    def test_copy_step_sets_transformed_data_and_splits(self):
        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
        )
        CopyStep().run(ctx)
        assert ctx.transformed_data is not None
        assert ctx.transformed_data is not container
        assert ctx.available_splits is not None
        assert "train" in ctx.available_splits
        assert ctx.transformed_results_for_new_key is not None
        assert ctx._raw_transformed_by_split == {}

    def test_copy_step_no_apply_to_keys_raises(self):
        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=[],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
        )
        with pytest.raises(ValueError, match="apply_to_keys must not be empty"):
            CopyStep().run(ctx)

    def test_copy_step_missing_apply_key_raises(self):
        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["nonexistent"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
        )
        with pytest.raises(KeyError, match="not found in data"):
            CopyStep().run(ctx)

    def test_copy_step_transform_on_keys_empty_intersection_raises(self):
        """When transform_on_keys has no overlap with data splits, CopyStep raises ValueError."""
        container = create_dummy_single_unit_container()  # has "train", "val"
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
            transform_on_keys=["valid", "test"],  # no overlap with train, val
        )
        with pytest.raises(ValueError) as exc_info:
            CopyStep().run(ctx)
        msg = str(exc_info.value)
        # Print full error in debug (run with: pytest -s ...)
        print(
            "\n--- ValueError (transform_on_keys empty intersection) ---\n"
            + msg
            + "\n---"
        )
        assert "no splits" in msg and "transform_on_keys" in msg
        assert "train" in msg and "val" in msg
        assert "valid" in msg and "test" in msg


class TestHooks:
    """Lifecycle hook registry and emit."""

    def test_register_and_emit(self):
        clear_hooks()
        seen = []

        def cb(event, context):
            seen.append((event, context.data is not None))

        register_hook("on_pipeline_start", cb)
        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
        )
        _emit("on_pipeline_start", ctx)
        assert len(seen) == 1
        assert seen[0][0] == "on_pipeline_start"
        assert seen[0][1] is True
        clear_hooks()

    def test_clear_hooks_event(self):
        clear_hooks()
        seen = []

        def cb(event, context):
            seen.append(event)

        register_hook("on_pipeline_end", cb)
        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=None,
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
        )
        _emit("on_pipeline_end", ctx)
        assert len(seen) == 1
        clear_hooks("on_pipeline_end")
        seen.clear()
        _emit("on_pipeline_end", ctx)
        assert len(seen) == 0
        clear_hooks()

    def test_hook_exception_does_not_break_emit(self):
        clear_hooks()

        def bad_cb(event, context):
            raise RuntimeError("hook failed")

        def good_cb(event, context):
            good_cb.called = True

        good_cb.called = False
        register_hook("before_transform", bad_cb)
        register_hook("before_transform", good_cb)
        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=None,
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
        )
        _emit("before_transform", ctx)
        assert good_cb.called is True
        clear_hooks()


class TestStrategyViaPipeline:
    """Strategy.apply uses pipeline; behaviour unchanged (smoke)."""

    def test_strategy_apply_returns_tuple_via_pipeline(self):
        from picid.transforms.base.strategy import TransformStrategy

        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = create_dummy_single_unit_container()
        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )
        assert result is not None
        assert isinstance(log, dict)
        assert "features" in result

    def test_fittable_transform_without_fit_on_raises(self):
        """When transform has requires_fit=True but fit_on_split/fit_on_key are not set, pipeline raises."""
        from picid.exceptions import TransformError
        from picid.transforms.base.strategy import TransformStrategy

        strategy = TransformStrategy()
        transform = (
            DummyFittableTransform()
        )  # has ConcatFitAndPerSegmentTransformMixin -> requires_fit=True
        container = create_dummy_single_unit_container()
        with pytest.raises(TransformError) as exc_info:
            strategy.apply(
                transform_instance=transform,
                data=container,
                apply_to_keys="features",
                assign_to_keys="features",
                assign_to_keys_map=["features"],
                fit_on_split=None,
                fit_on_key=None,
            )
        assert isinstance(exc_info.value.cause, ValueError)
        msg = str(exc_info.value)
        assert "requires fitting" in msg
        assert "fit_on" in msg
