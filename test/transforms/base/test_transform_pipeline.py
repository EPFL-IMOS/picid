"""Tests for picid.transforms.base.transform_pipeline."""

import numpy as np
import pytest

from picid.transforms.base.transform_pipeline import (
    TransformPipeline,
    TransformSequenceProtocol,
)
from picid.transforms.base.data_transform import DataTransform

from test.transforms.base.conftest import (
    DummyStatelessTransform,
    create_dummy_split_container,
)


class TestTransformPipelineInit:
    def test_init_valid(self):
        t = DummyStatelessTransform()
        dt1 = DataTransform(
            "scale", t, {"apply_to": "features", "assign_to": "features"}
        )
        pipeline = TransformPipeline([dt1])
        assert len(pipeline) == 1
        assert "scale" in pipeline

    def test_init_non_datatransform_raises(self):
        with pytest.raises(ValueError, match="must be DataTransform"):
            TransformPipeline([DummyStatelessTransform()])

    def test_init_duplicate_names_raises(self):
        t = DummyStatelessTransform()
        dt1 = DataTransform("same", t, {"apply_to": "features"})
        dt2 = DataTransform("same", t, {"apply_to": "target"})
        with pytest.raises(ValueError, match="must be unique|Duplicates"):
            TransformPipeline([dt1, dt2])


class TestTransformPipelineRun:
    def test_run_applies_transforms(self):
        t = DummyStatelessTransform()
        dt = DataTransform(
            "double", t, {"apply_to": "features", "assign_to": "features"}
        )
        pipeline = TransformPipeline([dt])
        data = create_dummy_split_container(
            n_units=1, n_samples_per_unit=3, n_features=2
        )
        orig = data["features"]["train"][0].copy()
        result = pipeline.run(data)
        # DummyStatelessTransform multiplies by 2
        expected = orig * 2
        np.testing.assert_array_almost_equal(
            result["features"]["train"][0],
            expected,
        )

    def test_run_multiple_transforms_sequential(self):
        t = DummyStatelessTransform()
        dt1 = DataTransform("t1", t, {"apply_to": "features", "assign_to": "features"})
        dt2 = DataTransform("t2", t, {"apply_to": "features", "assign_to": "features"})
        pipeline = TransformPipeline([dt1, dt2])
        data = create_dummy_split_container(
            n_units=1, n_samples_per_unit=2, n_features=2
        )
        orig = data["features"]["train"][0].copy()
        result = pipeline.run(data)
        # 2x * 2x = 4x
        expected = orig * 4
        np.testing.assert_array_almost_equal(
            result["features"]["train"][0],
            expected,
        )


class TestTransformPipelineProtocol:
    def test_config_property(self):
        t = DummyStatelessTransform()
        dt = DataTransform("x", t, {"apply_to": "features"})
        pipeline = TransformPipeline([dt])
        cfg = pipeline.config
        assert isinstance(cfg, dict)
        assert "x" in cfg
        assert "transform_class" in cfg["x"]
        assert cfg["x"]["transform_class"] == "DummyStatelessTransform"
        assert "metadata" in cfg["x"]

    def test_get_transforms(self):
        t = DummyStatelessTransform()
        dt = DataTransform("a", t, {"apply_to": "features"})
        pipeline = TransformPipeline([dt])
        transforms = pipeline.get_transforms()
        assert "a" in transforms
        assert transforms["a"] is dt

    def test_get_cache_point_names(self):
        t = DummyStatelessTransform()
        dt1 = DataTransform("t1", t, {"apply_to": "features", "cache_point": True})
        dt2 = DataTransform("t2", t, {"apply_to": "features"})
        pipeline = TransformPipeline([dt1, dt2])
        names = pipeline.get_cache_point_names()
        assert names == ["t1"]

    def test_get_transform_names_after(self):
        t = DummyStatelessTransform()
        dt1 = DataTransform("a", t, {"apply_to": "features"})
        dt2 = DataTransform("b", t, {"apply_to": "features"})
        dt3 = DataTransform("c", t, {"apply_to": "features"})
        pipeline = TransformPipeline([dt1, dt2, dt3])
        assert pipeline.get_transform_names_after("a") == ["b", "c"]
        assert pipeline.get_transform_names_after("b") == ["c"]
        assert pipeline.get_transform_names_after("c") == []
        # Unknown name returns full list (fallback)
        after_unknown = pipeline.get_transform_names_after("unknown")
        assert after_unknown == ["a", "b", "c"]

    def test_get_config_up_to_and_including(self):
        t = DummyStatelessTransform()
        dt1 = DataTransform("a", t, {"apply_to": "features"})
        dt2 = DataTransform("b", t, {"apply_to": "features"})
        pipeline = TransformPipeline([dt1, dt2])
        cfg = pipeline.get_config_up_to_and_including("a")
        assert "a" in cfg
        assert "b" not in cfg
        cfg2 = pipeline.get_config_up_to_and_including("b")
        assert "a" in cfg2
        assert "b" in cfg2

    def test_len_and_contains(self):
        t = DummyStatelessTransform()
        dt1 = DataTransform("x", t, {"apply_to": "features"})
        pipeline = TransformPipeline([dt1])
        assert len(pipeline) == 1
        assert "x" in pipeline
        assert "y" not in pipeline

    def test_repr(self):
        t = DummyStatelessTransform()
        dt = DataTransform("scale", t, {"apply_to": "features"})
        pipeline = TransformPipeline([dt])
        r = repr(pipeline)
        assert "TransformPipeline" in r
        assert "scale" in r


class TestTransformSequenceProtocol:
    def test_pipeline_satisfies_protocol(self):
        """TransformPipeline is runtime checkable as TransformSequenceProtocol."""
        t = DummyStatelessTransform()
        dt = DataTransform("x", t, {"apply_to": "features"})
        pipeline = TransformPipeline([dt])
        assert isinstance(pipeline, TransformSequenceProtocol)
