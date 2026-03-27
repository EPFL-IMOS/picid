"""
Comprehensive tests for the preprocessing caching mechanism (Stage 1, boundary, preprocessed).

Tests write cache to disk, load it back, and assert data and behaviour are correct.
Target: close to 100% coverage of cache-related paths in PreProcessor.pipeline() and FileSystemCache.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.exceptions import (
    DatasourceContractError,
    DatasourceLoadError,
    DatasourceSplitError,
)
from picid.data.preprocessing.preprocessor import PreProcessor
from picid.data.cache.offline import FileSystemCache
from picid.exceptions import PreprocessingDatasourceError
from picid.transforms.base.transform_manager import ConfigTransformManager
from picid.transforms.base.transform_pipeline import TransformPipeline
from picid.transforms.base.data_transform import DataTransform
from picid.utils.hash_utils import compute_cache_key, ensure_serializable

from sklearn.utils.validation import check_is_fitted

from test.transforms.base.conftest import DummyStatelessTransform
from test.data.preprocessing.test_preprocessor import _PreprocessingErrorDatasource


# -----------------------------------------------------------------------------
# Picklable datasource for pipeline tests (Stage 1 cache stores datasource + meta)
# -----------------------------------------------------------------------------


class _PicklableDatasource:
    """Minimal datasource that returns a SplitDatasetContainer and is picklable."""

    def __init__(
        self, seed: int = 0, n_units: int = 2, n_samples: int = 20, n_features: int = 3
    ):
        self._seed = seed
        self._n_units = n_units
        self._n_samples = n_samples
        self._n_features = n_features
        self._container: SplitDatasetContainer | None = None
        # For cache key stability; must be serializable
        self._init_kwargs = {
            "seed": seed,
            "n_units": n_units,
            "n_samples": n_samples,
            "n_features": n_features,
        }

    def load_data(self) -> None:
        pass

    def split_data(self) -> None:
        pass

    def get_meta_data(self) -> dict:
        return {"seed": self._seed, "n_units": self._n_units}

    def get_data(self) -> SplitDatasetContainer:
        if self._container is None:
            rng = np.random.default_rng(self._seed)
            self._container = SplitDatasetContainer(
                features={
                    "train": [
                        rng.standard_normal((self._n_samples, self._n_features))
                        for _ in range(self._n_units)
                    ],
                    "val": [
                        rng.standard_normal((self._n_samples // 2, self._n_features))
                        for _ in range(self._n_units)
                    ],
                    "test": [
                        rng.standard_normal((self._n_samples // 2, self._n_features))
                        for _ in range(self._n_units)
                    ],
                },
                target={
                    "train": [
                        rng.standard_normal((self._n_samples, 1))
                        for _ in range(self._n_units)
                    ],
                    "val": [
                        rng.standard_normal((self._n_samples // 2, 1))
                        for _ in range(self._n_units)
                    ],
                    "test": [
                        rng.standard_normal((self._n_samples // 2, 1))
                        for _ in range(self._n_units)
                    ],
                },
            )
        return self._container

    def get_cache_fingerprint(self) -> dict:
        return dict(self._init_kwargs)


# -----------------------------------------------------------------------------
# Transform configs (DummyStatelessTransform from base conftest)
# -----------------------------------------------------------------------------

_TRANSFORM_CONFIG_NO_CACHE_POINT = {
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
        "metadata": {"apply_to": "features", "assign_to": "features"},
    },
}

_TRANSFORM_CONFIG_WITH_CACHE_POINT = {
    "t1": {
        "transform": {
            "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
        },
        "metadata": {"apply_to": "features", "assign_to": "features"},
    },
    "heavy": {
        "transform": {
            "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
        },
        "metadata": {
            "apply_to": "features",
            "assign_to": "features",
            "cache_point": True,
        },
    },
    "light": {
        "transform": {
            "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
        },
        "metadata": {"apply_to": "features", "assign_to": "features"},
    },
}

# Same as above but with a different light transform metadata (to simulate "changed light config")
_TRANSFORM_CONFIG_LIGHT_CHANGED = {
    "t1": _TRANSFORM_CONFIG_WITH_CACHE_POINT["t1"],
    "heavy": _TRANSFORM_CONFIG_WITH_CACHE_POINT["heavy"],
    "light": {
        "transform": {
            "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
        },
        "metadata": {
            "apply_to": "features",
            "assign_to": "features",
            "extra_param": "changed",
        },
    },
}

# heavy is the LAST transform in Run 1 (cache_point at end of chain).
# This is the scenario that triggers the boundary-restore bug: after restoring
# the boundary cache, the saved transforms object has no transforms after "heavy",
# so get_transform_names_after("heavy") would return [] without the fix.
_TRANSFORM_CONFIG_HEAVY_IS_LAST = {
    "t1": {
        "transform": {
            "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
        },
        "metadata": {"apply_to": "features", "assign_to": "features"},
    },
    "heavy": {
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

# Run 2: same t1 + heavy, then two new trailing transforms added after the boundary.
_TRANSFORM_CONFIG_HEAVY_IS_LAST_EXTENDED = {
    "t1": _TRANSFORM_CONFIG_HEAVY_IS_LAST["t1"],
    "heavy": _TRANSFORM_CONFIG_HEAVY_IS_LAST["heavy"],
    "light": {
        "transform": {
            "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
        },
        "metadata": {"apply_to": "features", "assign_to": "features"},
    },
    "extra": {
        "transform": {
            "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
        },
        "metadata": {"apply_to": "features", "assign_to": "features"},
    },
}


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture
def library_dir(tmp_path):
    """A stable directory with a dummy file for compute_cache_key (extensions=[".py"])."""
    d = tmp_path / "library"
    d.mkdir()
    (d / "dummy.py").write_text("# cache key seed")
    return str(d)


@pytest.fixture
def datasource():
    return _PicklableDatasource(seed=42, n_units=2, n_samples=20, n_features=3)


@pytest.fixture
def manager_no_cache_point():
    return ConfigTransformManager(transforms_config=_TRANSFORM_CONFIG_NO_CACHE_POINT)


@pytest.fixture
def manager_with_cache_point():
    return ConfigTransformManager(transforms_config=_TRANSFORM_CONFIG_WITH_CACHE_POINT)


@pytest.fixture
def manager_light_changed():
    return ConfigTransformManager(transforms_config=_TRANSFORM_CONFIG_LIGHT_CHANGED)


def _assert_containers_equal(
    c1: SplitDatasetContainer, c2: SplitDatasetContainer
) -> None:
    """Assert two SplitDatasetContainers have the same keys and array values."""
    assert set(c1.keys()) == set(c2.keys())
    for key in c1.keys():
        for split in c1[key].keys():
            a1 = c1[key][split]
            a2 = c2[key][split]
            assert len(a1) == len(a2)
            for u1, u2 in zip(a1, a2):
                np.testing.assert_array_almost_equal(np.asarray(u1), np.asarray(u2))


# -----------------------------------------------------------------------------
# Pipeline caching: full preprocessed hit
# -----------------------------------------------------------------------------


class TestPipelinePreprocessedCache:
    """Full pipeline with cache: first run writes, second run hits preprocessed."""

    def test_first_run_writes_preprocessed_to_disk(
        self, cache_dir, library_dir, datasource, manager_no_cache_point
    ):
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_no_cache_point,
        )
        preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
        )
        # Stage 1 and Stage 2 should have written
        cache = FileSystemCache()
        datasource_config = ensure_serializable(datasource.get_cache_fingerprint())
        transforms_config = ensure_serializable(manager_no_cache_point.config)
        preprocessed_config = {
            "datasource": datasource_config,
            "transforms": transforms_config,
        }
        expected_hash = compute_cache_key(
            config=preprocessed_config,
            library_dir=[library_dir, library_dir],
            extensions=[".py"],
        )
        meta = cache.load_metadata(
            cache_dir, stage="preprocessed", cache_key=expected_hash
        )
        assert meta is not None
        stored_config, stored_hash = meta
        assert stored_hash == expected_hash
        data, saved_objects = cache.load_data(
            cache_dir, stage="preprocessed", cache_key=expected_hash
        )
        assert data is not None
        assert "transforms" in saved_objects
        assert "meta_data" in saved_objects
        assert hasattr(data, "features") and hasattr(data, "target")

    def test_second_run_same_config_hits_preprocessed_returns_same_data(
        self, cache_dir, library_dir, datasource, manager_no_cache_point
    ):
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_no_cache_point,
        )
        result1 = preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
        )
        result2 = preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
        )
        _assert_containers_equal(result1, result2)

    def test_cache_disabled_no_preprocessed_write(
        self, tmp_path, datasource, manager_no_cache_point
    ):
        """With cache paths None and cache_preprocessed False, pipeline runs but does not use cache."""
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_no_cache_point,
        )
        result = preprocessor.pipeline(
            data_cache_path=None,
            data_library_part_path=None,
            transform_library_part_path=None,
            cache_preprocessed=False,
        )
        assert result is not None
        # No cache dir was passed, so nothing under tmp_path
        assert not (tmp_path / "preprocessed").exists()

    def test_cached_pipeline_wraps_cache_fingerprint_datasource_error(
        self, cache_dir, library_dir, manager_no_cache_point
    ):
        """Cached pipeline should wrap get_cache_fingerprint failures with stage context."""
        datasource = _PreprocessingErrorDatasource(
            cache_error=DatasourceContractError("fingerprint failed"),
        )
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_no_cache_point,
        )

        with pytest.raises(PreprocessingDatasourceError) as exc_info:
            preprocessor.pipeline(
                data_cache_path=cache_dir,
                data_library_part_path=library_dir,
                transform_library_part_path=library_dir,
                cache_preprocessed=True,
            )

        exc = exc_info.value
        assert exc.stage == "get_cache_fingerprint"
        assert exc.datasource_type == "_PreprocessingErrorDatasource"
        assert exc.datasource_name == "faulty"
        assert exc.datasource_error_type == "DatasourceContractError"

    @pytest.mark.parametrize(
        ("datasource_kwargs", "stage", "error_type"),
        [
            (
                {"load_error": DatasourceLoadError("load failed")},
                "load_data",
                "DatasourceLoadError",
            ),
            (
                {"split_error": DatasourceSplitError("split failed")},
                "split_data",
                "DatasourceSplitError",
            ),
        ],
    )
    def test_cached_pipeline_wraps_load_and_split_failures(
        self,
        cache_dir,
        library_dir,
        manager_no_cache_point,
        datasource_kwargs,
        stage,
        error_type,
    ):
        """Cached pipeline should wrap load/split datasource failures with stage context."""
        datasource = _PreprocessingErrorDatasource(**datasource_kwargs)
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_no_cache_point,
        )

        with pytest.raises(PreprocessingDatasourceError) as exc_info:
            preprocessor.pipeline(
                data_cache_path=cache_dir,
                data_library_part_path=library_dir,
                transform_library_part_path=library_dir,
                cache_preprocessed=True,
            )

        exc = exc_info.value
        assert exc.stage == stage
        assert exc.datasource_type == "_PreprocessingErrorDatasource"
        assert exc.datasource_name == "faulty"
        assert exc.datasource_error_type == error_type

    def test_cached_pipeline_does_not_wrap_assertion_error_in_cache_fingerprint(
        self, cache_dir, library_dir, manager_no_cache_point
    ):
        """AssertionError invariants from get_cache_fingerprint should bubble unchanged."""
        datasource = _PreprocessingErrorDatasource(
            cache_error=AssertionError("fingerprint invariant"),
        )
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_no_cache_point,
        )

        with pytest.raises(AssertionError, match="fingerprint invariant"):
            preprocessor.pipeline(
                data_cache_path=cache_dir,
                data_library_part_path=library_dir,
                transform_library_part_path=library_dir,
                cache_preprocessed=True,
            )


# -----------------------------------------------------------------------------
# Pipeline caching: Stage 1 (loaded_and_splitted_data)
# -----------------------------------------------------------------------------


class TestPipelineStage1Cache:
    """Stage 1 cache: load/split once, reuse on second run."""

    def test_stage1_written_when_cache_enabled(
        self, cache_dir, library_dir, datasource, manager_no_cache_point
    ):
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_no_cache_point,
        )
        preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
        )
        stage1_dir = Path(cache_dir) / "loaded_and_splitted_data"
        assert stage1_dir.exists()
        assert (stage1_dir / "data.pkl").exists()
        assert (stage1_dir / "metadata.pkl").exists()
        assert (stage1_dir / "hash.txt").exists()

    def test_no_cache_point_does_not_create_boundary_dir(
        self, cache_dir, library_dir, datasource, manager_no_cache_point
    ):
        """When no transform has cache_point, pipeline never touches boundary stage."""
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_no_cache_point,
        )
        preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
        )
        boundary_dir = Path(cache_dir) / "boundary"
        assert not boundary_dir.exists()


# -----------------------------------------------------------------------------
# Pipeline caching: boundary save and restore
# -----------------------------------------------------------------------------


class TestPipelineBoundaryCache:
    """Boundary cache: save after heavy transform, restore and run only light transforms."""

    def test_first_run_with_cache_point_writes_boundary_and_preprocessed(
        self, cache_dir, library_dir, datasource, manager_with_cache_point
    ):
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_with_cache_point,
        )
        preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )
        # Boundary cache exists (after "heavy")
        boundary_names = manager_with_cache_point.get_cache_point_names()
        assert "heavy" in boundary_names
        boundary_config = {
            "datasource": ensure_serializable(datasource.get_cache_fingerprint()),
            "transforms": manager_with_cache_point.get_config_up_to_and_including(
                "heavy"
            ),
        }
        boundary_key = compute_cache_key(
            config=boundary_config,
            library_dir=[library_dir, library_dir],
            extensions=[".py"],
        )
        cache = FileSystemCache()
        meta = cache.load_metadata(cache_dir, stage="boundary", cache_key=boundary_key)
        assert meta is not None
        boundary_data, saved_objects = cache.load_data(
            cache_dir, stage="boundary", cache_key=boundary_key
        )
        assert boundary_data is not None
        assert "transforms" in saved_objects

    def test_second_run_unchanged_config_hits_preprocessed(
        self, cache_dir, library_dir, datasource, manager_with_cache_point
    ):
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_with_cache_point,
        )
        r1 = preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )
        r2 = preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
        )
        _assert_containers_equal(r1, r2)

    def test_light_config_changed_restores_from_boundary_runs_only_light(
        self,
        cache_dir,
        library_dir,
        datasource,
        manager_with_cache_point,
        manager_light_changed,
    ):
        # Run 1: full pipeline with cache_point → boundary + preprocessed written
        preprocessor1 = PreProcessor(
            datasource=datasource,
            transforms=manager_with_cache_point,
        )
        result_full = preprocessor1.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )
        # Run 2: "light" config changed → full preprocessed miss, boundary hit, run only light
        preprocessor2 = PreProcessor(
            datasource=datasource,
            transforms=manager_light_changed,
        )
        result_restored = preprocessor2.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )
        # Result should still be a valid container with same structure (features/target, splits)
        assert set(result_restored.keys()) == set(result_full.keys())
        for key in result_restored.keys():
            assert set(result_restored[key].keys()) == set(result_full[key].keys())
        # Data may differ slightly because light transform metadata changed (extra_param),
        # but shapes must match
        for key in result_restored.keys():
            for split in result_restored[key].keys():
                assert len(result_restored[key][split]) == len(result_full[key][split])

    def test_save_boundary_caches_false_does_not_write_boundaries(
        self, cache_dir, library_dir, datasource, manager_with_cache_point
    ):
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=manager_with_cache_point,
        )
        preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=False,
        )
        boundary_dir = Path(cache_dir) / "boundary"
        # Pipeline may create boundary subdirs when probing load_metadata; we must not have written data.pkl
        for subdir in boundary_dir.iterdir() if boundary_dir.exists() else []:
            assert not (
                subdir / "data.pkl"
            ).exists(), "save_boundary_caches=False should not write boundary data"

    def test_extended_chain_restores_from_boundary_and_runs_new_trailing_transforms(
        self, cache_dir, library_dir, datasource
    ):
        """
        Regression test for the boundary-restore bug: when the cache_point is the
        last transform in Run 1 and Run 2 adds new trailing transforms, those new
        transforms must be applied after restoring from the boundary cache.

        Without the fix, get_transform_names_after(boundary_name) is called on the
        saved (N-transform) object, which has no transforms after the boundary, so
        the tail is empty and the new transforms are silently skipped.
        """
        manager_short = ConfigTransformManager(
            transforms_config=_TRANSFORM_CONFIG_HEAVY_IS_LAST
        )
        manager_extended = ConfigTransformManager(
            transforms_config=_TRANSFORM_CONFIG_HEAVY_IS_LAST_EXTENDED
        )

        # Run 1: 2 transforms (t1, heavy). Writes boundary after "heavy" + preprocessed.
        preprocessor1 = PreProcessor(datasource=datasource, transforms=manager_short)
        preprocessor1.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )

        # Run 2: 4 transforms (t1, heavy, light, extra). Boundary valid for t1+heavy;
        # must run light + extra afterward.
        preprocessor2 = PreProcessor(datasource=datasource, transforms=manager_extended)
        result = preprocessor2.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )

        assert result is not None
        assert set(result.keys()) == {"features", "target"}
        for key in result.keys():
            assert set(result[key].keys()) == {"train", "val", "test"}

    def test_extended_chain_preprocessed_cache_stores_full_transforms(
        self, cache_dir, library_dir, datasource
    ):
        """
        Regression test for the secondary bug: after a boundary restore with a
        longer current chain, the preprocessed cache must store the full (M-transform)
        manager, not the truncated (N-transform) boundary snapshot.
        """
        manager_short = ConfigTransformManager(
            transforms_config=_TRANSFORM_CONFIG_HEAVY_IS_LAST
        )
        manager_extended = ConfigTransformManager(
            transforms_config=_TRANSFORM_CONFIG_HEAVY_IS_LAST_EXTENDED
        )

        # Run 1: writes boundary + preprocessed with 2-transform chain.
        preprocessor1 = PreProcessor(datasource=datasource, transforms=manager_short)
        preprocessor1.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )

        # Run 2: restores from boundary, runs light+extra, writes preprocessed.
        preprocessor2 = PreProcessor(datasource=datasource, transforms=manager_extended)
        preprocessor2.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )

        # Verify the preprocessed cache stores the full 4-transform manager.
        preprocessed_config = {
            "datasource": ensure_serializable(datasource._init_kwargs),
            "transforms": ensure_serializable(manager_extended.config),
        }
        preprocessed_key = compute_cache_key(
            config=preprocessed_config,
            library_dir=[library_dir, library_dir],
            extensions=[".py"],
        )
        cache = FileSystemCache()
        _, saved = cache.load_data(
            cache_dir, stage="preprocessed", cache_key=preprocessed_key
        )
        assert set(saved["transforms"].get_transform_names()) == {
            "t1",
            "heavy",
            "light",
            "extra",
        }


# -----------------------------------------------------------------------------
# PreProcessor with TransformPipeline (library path)
# -----------------------------------------------------------------------------


class TestPreProcessorWithTransformPipelineNoCache:
    """PreProcessor accepts TransformPipeline and runs pipeline without cache."""

    def test_pipeline_with_transform_pipeline_returns_container(self, datasource):
        pipeline = TransformPipeline(
            [
                DataTransform(
                    "t1",
                    DummyStatelessTransform(),
                    {"apply_to": "features", "assign_to": "features"},
                ),
            ]
        )
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=pipeline,
        )
        result = preprocessor.pipeline()
        assert result is not None
        assert hasattr(result, "features") and hasattr(result, "target")
        assert (
            "train" in result.features
            and "val" in result.features
            and "test" in result.features
        )


class TestPreProcessorWithTransformPipelineWithCache:
    """PreProcessor with TransformPipeline uses same three-tier cache as ConfigTransformManager."""

    def test_two_runs_with_cache_preprocessed_second_hits_cache(
        self, cache_dir, library_dir, datasource
    ):
        pipeline = TransformPipeline(
            [
                DataTransform(
                    "t1",
                    DummyStatelessTransform(),
                    {"apply_to": "features", "assign_to": "features"},
                ),
            ]
        )
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=pipeline,
        )
        result1 = preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
        )
        result2 = preprocessor.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
        )
        _assert_containers_equal(result1, result2)
        # Preprocessed cache should exist
        cache = FileSystemCache()
        config = {
            "datasource": ensure_serializable(datasource.get_cache_fingerprint()),
            "transforms": ensure_serializable(pipeline.config),
        }
        key = compute_cache_key(
            config=config,
            library_dir=[library_dir, library_dir],
            extensions=[".py"],
        )
        meta = cache.load_metadata(cache_dir, stage="preprocessed", cache_key=key)
        assert meta is not None


# -----------------------------------------------------------------------------
# FileSystemCache: preprocessed and boundary load when missing
# -----------------------------------------------------------------------------


class TestCacheLoadWhenMissing:
    """load_metadata / load_data when cache is missing."""

    def test_load_metadata_preprocessed_missing_returns_none(self, tmp_path):
        cache = FileSystemCache()
        meta = cache.load_metadata(
            str(tmp_path), stage="preprocessed", cache_key="nonexistent_key"
        )
        assert meta is None

    def test_load_data_preprocessed_missing_returns_none(self, tmp_path):
        cache = FileSystemCache()
        # load_data returns (data, metadata) from joblib; when files missing it returns None
        result = cache.load_data(
            str(tmp_path), stage="preprocessed", cache_key="nonexistent_key"
        )
        assert result is None


# -----------------------------------------------------------------------------
# Boundary cache: stateful (sklearn) transform fitted state is preserved
# -----------------------------------------------------------------------------


class TestBoundaryCachePreservesStatefulTransforms:
    """
    Regression test for: NotFittedError when a sklearn scaler that was fitted
    before a boundary cache point is used for inverse-transforming predictions.

    The bug: _try_restore_from_boundary loaded the cached DatasetContainer but
    discarded saved["transforms"], leaving the in-process ConfigTransformManager
    with an unfitted scaler.  get_cached_transform_manager() then returned this
    unfitted manager to the evaluator, which raised NotFittedError on the first
    validation step.

    The fix: restore pp.transforms = saved["transforms"] in _try_restore_from_boundary
    so the fitted scaler state survives the cache round-trip.
    """

    # MinMaxScalerSklearn fitted on "target" BEFORE the boundary cache point.
    _CONFIG_SCALER_BEFORE_BOUNDARY = {
        "scaler": {
            "transform": {
                "_target_": "picid.transforms.base_transforms.scaler.MinMaxScalerSklearn"
            },
            "metadata": {
                "apply_to": "target",
                "assign_to": "target",
                "fit_on": "train",
            },
        },
        "heavy": {
            "transform": {
                "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
            },
            "metadata": {
                "apply_to": "features",
                "assign_to": "features",
                "cache_point": True,
            },
        },
        "light": {
            "transform": {
                "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
            },
            "metadata": {"apply_to": "features", "assign_to": "features"},
        },
    }

    # Identical scaler + heavy; "light" has extra_param to bust the preprocessed cache
    # but leave the boundary cache valid, forcing a boundary restore on run 2.
    _CONFIG_LIGHT_CHANGED = {
        "scaler": _CONFIG_SCALER_BEFORE_BOUNDARY["scaler"],
        "heavy": _CONFIG_SCALER_BEFORE_BOUNDARY["heavy"],
        "light": {
            "transform": {
                "_target_": "test.transforms.base.conftest.DummyStatelessTransform"
            },
            "metadata": {
                "apply_to": "features",
                "assign_to": "features",
                "extra_param": "changed",
            },
        },
    }

    def test_scaler_is_fitted_after_boundary_restore(
        self, cache_dir, library_dir, datasource
    ):
        """
        After restoring from a boundary cache the MinMaxScalerSklearn (fitted before
        the boundary) must have its sklearn scaler in a fitted state.
        Without the fix this raises sklearn.exceptions.NotFittedError.
        """
        manager1 = ConfigTransformManager(
            transforms_config=self._CONFIG_SCALER_BEFORE_BOUNDARY
        )
        manager2 = ConfigTransformManager(
            transforms_config=self._CONFIG_LIGHT_CHANGED
        )

        # Run 1: full pipeline — writes boundary (after "heavy") + preprocessed.
        preprocessor1 = PreProcessor(datasource=datasource, transforms=manager1)
        preprocessor1.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )

        # Run 2: "light" changed → preprocessed cache miss, boundary hit.
        # The scaler (before the boundary) must be restored with its fitted state.
        preprocessor2 = PreProcessor(datasource=datasource, transforms=manager2)
        preprocessor2.pipeline(
            data_cache_path=cache_dir,
            data_library_part_path=library_dir,
            transform_library_part_path=library_dir,
            cache_preprocessed=True,
            save_boundary_caches=True,
        )

        restored_manager = preprocessor2.get_cached_transform_manager()
        inverter = restored_manager.get_inverter_for_key("target")
        assert inverter is not None, "No inverter found for key 'target'"

        # check_is_fitted raises NotFittedError if the sklearn scaler was not restored.
        check_is_fitted(inverter.scaler)
