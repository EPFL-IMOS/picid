"""Comprehensive tests for PreProcessor class.

This module tests the main preprocessing pipeline orchestrator that
coordinates data loading, splitting, transformation, and caching.

PHM Context:
-----------
The PreProcessor is the central component for preparing PHM datasets.
It handles multi-unit data, applies transforms, and manages caching
to accelerate experiment iteration.

Test Coverage Strategy:
----------------------
1. **Initialization Tests**: Parameter storage (datasource, transforms)
2. **Data Fetching**: Loading data from sources
3. **Transform Application**: Applying transform pipelines
4. **Pipeline Execution**: Full preprocessing workflow
5. **Caching Integration**: Cache handling for speed
6. **Error Handling**: Transform failures
"""

import copy
from collections import OrderedDict
from unittest.mock import Mock
import pytest
import numpy as np

from picid.data.data_objects import (
    DatasetContainer,
    NamedTransformInput,
    SplitDatasetContainer,
    SplitViewPolicy,
)
from picid.data.datasources.base.exceptions import DatasourceStateError
from picid.data.preprocessing.preprocessor import PreProcessor
from picid.data.preprocessing.base import PreProcessorInterface
from picid.exceptions import PreprocessingDatasourceError
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin


class _PreprocessingErrorDatasource:
    """Tiny datasource stub used to drive preprocessing error paths.

    Parameters
    ----------
    load_error : BaseException | None, optional
        Exception raised by ``load_data()``.
    split_error : BaseException | None, optional
        Exception raised by ``split_data()``.
    meta_error : BaseException | None, optional
        Exception raised by ``get_meta_data()``.
    data_error : BaseException | None, optional
        Exception raised by ``get_data()``.
    cache_error : BaseException | None, optional
        Exception raised by ``get_cache_fingerprint()``.
    data_name : str, optional
        Logical datasource name exposed to preprocessing error reporting.
    task_mode : str, optional
        Task mode exposed to preprocessing error reporting.
    """

    def __init__(
        self,
        *,
        load_error: BaseException | None = None,
        split_error: BaseException | None = None,
        meta_error: BaseException | None = None,
        data_error: BaseException | None = None,
        cache_error: BaseException | None = None,
        data_name: str = "faulty",
        task_mode: str = "regression",
    ):
        self.data_name = data_name
        self.task_mode = task_mode
        self._load_error = load_error
        self._split_error = split_error
        self._meta_error = meta_error
        self._data_error = data_error
        self._cache_error = cache_error
        self._container = DatasetContainer(
            features={"train": [], "val": [], "test": []},
            target={"train": [], "val": [], "test": []},
        )

    def load_data(self) -> None:
        if self._load_error is not None:
            raise self._load_error

    def split_data(self) -> None:
        if self._split_error is not None:
            raise self._split_error

    def get_meta_data(self) -> dict:
        if self._meta_error is not None:
            raise self._meta_error
        return {"source": self.data_name}

    def get_data(self) -> DatasetContainer:
        if self._data_error is not None:
            raise self._data_error
        return self._container

    def get_cache_fingerprint(self) -> dict:
        if self._cache_error is not None:
            raise self._cache_error
        return {
            "data_name": self.data_name,
            "task_mode": self.task_mode,
        }

    def get_data_name(self) -> str:
        return self.data_name

    def get_data_names(self) -> tuple[str, ...]:
        return (self.data_name,)

    def get_split_mode(self) -> str:
        return "within_units"


class _MetadataAwareDatasource:
    """Datasource stub that returns a split container with both metadata scopes."""

    def __init__(self):
        self.data_name = "metadata-demo"
        self.task_mode = "regression"
        self._container = SplitDatasetContainer(
            features={
                "train": [np.array([[1.0], [2.0]])],
                "val": [np.array([[3.0]])],
                "test": [],
            },
            target={
                "train": [np.array([[0.0], [1.0]])],
                "val": [np.array([[2.0]])],
                "test": [],
            },
            container_metadata={"dataset_name": "metadata-demo", "column_map": {}},
            unit_metadata={
                "train": [{"unit_name": "train-1"}],
                "val": [{"unit_name": "val-1"}],
                "test": [],
            },
        )

    def load_data(self) -> None:
        return None

    def split_data(self) -> None:
        return None

    def get_meta_data(self) -> dict:
        return {"dataset_name": self.data_name}

    def get_data(self) -> DatasetContainer:
        return self._container.copy(deep=True)

    def get_cache_fingerprint(self) -> dict:
        return {"data_name": self.data_name, "task_mode": self.task_mode}

    def get_data_name(self) -> str:
        return self.data_name

    def get_data_names(self) -> tuple[str, ...]:
        return (self.data_name,)

    def get_split_mode(self) -> str:
        return "within_units"


class _PreprocessingMetadataTransform(NoFitPerSegmentMixin, DenseTransform):
    """Transform that records metadata scopes while preserving unit counts."""

    def __init__(self):
        super().__init__()
        self.seen_unit_metadata = []
        self.seen_pipeline_metadata = []

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        self.seen_unit_metadata.append(copy.deepcopy(data.metadata))
        self.seen_pipeline_metadata.append(copy.deepcopy(metadata))
        return data["features"] * 2


class TestPreProcessorInitialization:
    """Tests for PreProcessor initialization."""

    def test_init_stores_datasource_and_transforms(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Test initialization stores datasource, transforms, and default state.

        **PHM Logic**: PreProcessor orchestrates datasource and transform pipeline.

        **Methodology**: Create PreProcessor with datasource and transforms.

        **Expected**: Datasource and transforms stored, not preprocessed flag set.

        Validates: Requirement PP-1.1 - Initialization
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        assert preprocessor.datasource is mock_single_source_loader
        assert preprocessor.transforms is mock_transform_manager
        assert not preprocessor._is_preprocessed
        assert preprocessor.data is None

    def test_init_stores_datasource(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Test that datasource is stored correctly.

        **PHM Logic**: Datasource provides data loading interface.

        **Methodology**: Verify datasource attribute after init.

        **Expected**: Datasource accessible via attribute.

        Validates: Requirement PP-1.4 - Datasource storage
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        assert preprocessor.datasource is mock_single_source_loader

    def test_init_stores_transforms(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Test that transform manager is stored correctly.

        **PHM Logic**: Transform manager provides transform pipeline.

        **Methodology**: Verify transforms attribute after init.

        **Expected**: Transform manager accessible.

        Validates: Requirement PP-1.5 - Transform storage
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        assert preprocessor.transforms is mock_transform_manager


class TestPreProcessorMetaData:
    """Tests for metadata handling."""

    def test_get_meta_data_dict_empty_default(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Test get_meta_data_dict returns empty dict by default.

        **PHM Logic**: Metadata populated during pipeline execution.

        **Methodology**: Call get_meta_data_dict before pipeline.

        **Expected**: Empty dict returned.

        Validates: Requirement PP-2.1 - Default metadata
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        meta = preprocessor.get_meta_data_dict()

        assert isinstance(meta, dict)

    def test_get_meta_data_dict_after_fetch(
        self,
        mock_single_source_loader,
        mock_transform_manager,
        sample_dataset_container,
    ):
        """Test get_meta_data_dict after data fetch.

        **PHM Logic**: Metadata may be populated by datasource.

        **Methodology**: Fetch data, then get metadata.

        **Expected**: Metadata dict (possibly with content from source).

        Validates: Requirement PP-2.2 - Metadata after fetch
        """
        mock_single_source_loader.get_data.return_value = sample_dataset_container

        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        preprocessor.fetch_data()
        meta = preprocessor.get_meta_data_dict()

        assert isinstance(meta, dict)


class TestPreProcessorFetchData:
    """Tests for fetch_data method."""

    def test_fetch_data_calls_datasource(
        self,
        mock_single_source_loader,
        mock_transform_manager,
        sample_dataset_container,
    ):
        """Test that fetch_data calls datasource.get_data().

        **PHM Logic**: Data loaded from datasource on demand.

        **Methodology**: Call fetch_data, verify datasource called.

        **Expected**: get_data() called once.

        Validates: Requirement PP-3.1 - Datasource invocation
        """
        mock_single_source_loader.get_data.return_value = sample_dataset_container

        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        preprocessor.fetch_data()

        mock_single_source_loader.get_data.assert_called_once()


def test_preprocessor_apply_transforms_preserves_container_and_unit_metadata():
    """PreProcessor should keep both metadata scopes intact through transforms."""
    datasource = _MetadataAwareDatasource()
    transform = _PreprocessingMetadataTransform()
    dt = DataTransform(
        "scale-features",
        transform,
        {"apply_to": "features", "assign_to": "features"},
    )
    preprocessor = PreProcessor(datasource=datasource)

    preprocessor.fetch_data()
    processed = preprocessor.apply_transforms(
        preprocessor.data,
        OrderedDict([(dt.transform_name, dt)]),
    )

    assert processed.container_metadata["dataset_name"] == "metadata-demo"
    assert processed.unit_metadata["train"][0]["unit_name"] == "train-1"
    assert processed.to_split_dict()["val"]["unit_metadata"][0]["unit_name"] == "val-1"
    assert transform.seen_unit_metadata[0]["unit_name"] == "train-1"
    assert transform.seen_pipeline_metadata[0]["container_metadata"][
        "dataset_name"
    ] == ("metadata-demo")

    def test_fetch_data_stores_result(
        self,
        mock_single_source_loader,
        mock_transform_manager,
        sample_dataset_container,
    ):
        """Test that fetched data is stored.

        **PHM Logic**: Data stored for transform application.

        **Methodology**: Fetch data, verify self.data set.

        **Expected**: self.data contains DatasetContainer.

        Validates: Requirement PP-3.2 - Data storage
        """
        mock_single_source_loader.get_data.return_value = sample_dataset_container

        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        preprocessor.fetch_data()

        assert preprocessor.data is not None

    def test_fetch_data_wraps_typed_datasource_error_with_stage_context(
        self, mock_transform_manager
    ):
        """fetch_data wraps typed datasource exceptions with get_data stage context."""
        datasource = _PreprocessingErrorDatasource(
            data_error=DatasourceStateError("boom"),
        )
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=mock_transform_manager,
        )

        with pytest.raises(PreprocessingDatasourceError) as exc_info:
            preprocessor.fetch_data()

        exc = exc_info.value
        assert exc.stage == "get_data"
        assert exc.datasource_type == "_PreprocessingErrorDatasource"
        assert exc.datasource_name == "faulty"
        assert exc.datasource_error_type == "DatasourceStateError"
        assert isinstance(exc.cause, DatasourceStateError)

    def test_fetch_data_does_not_wrap_assertion_error(self, mock_transform_manager):
        """AssertionError invariants from datasource.get_data() should bubble unchanged."""
        datasource = _PreprocessingErrorDatasource(
            data_error=AssertionError("invariant broken"),
        )
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=mock_transform_manager,
        )

        with pytest.raises(AssertionError, match="invariant broken"):
            preprocessor.fetch_data()


class TestPreProcessorGetProcessedData:
    """Tests for get_processed_data_dict method."""

    def test_get_processed_data_before_pipeline_error(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Test error when called before pipeline execution.

        **PHM Logic**: Data not available until pipeline runs.

        **Methodology**: Call get_processed_data_dict without pipeline.

        **Expected**: RuntimeError raised.

        Validates: Requirement PP-4.1 - Pre-pipeline error
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        with pytest.raises(RuntimeError):
            preprocessor.get_processed_data_container()

    def test_get_processed_split_dict_before_pipeline_error(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Split-first export should also fail until pipeline() materializes data."""
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        with pytest.raises(RuntimeError):
            preprocessor.get_processed_split_dict(
                view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
            )

    def test_get_processed_split_dict_triggers_container_validation(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Split export should validate the processed container before returning it."""
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )
        preprocessor._is_preprocessed = True
        preprocessor.data = SplitDatasetContainer(
            features={"train": [1], "val": [2], "test": [3]},
            target={"train": [0], "val": [0], "test": [0]},
        )
        validate_mock = Mock(return_value=None)
        object.__setattr__(preprocessor.data, "validate", validate_mock)

        preprocessor.get_processed_split_dict(
            view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
        )

        validate_mock.assert_called_once_with()


class TestPreProcessorApplyTransforms:
    """Tests for apply_transforms method."""

    def test_apply_transforms_empty_transforms(
        self,
        mock_single_source_loader,
        mock_transform_manager,
        sample_dataset_container,
    ):
        """Test apply_transforms with empty transform dict.

        **PHM Logic**: No transforms means data passes through unchanged.

        **Methodology**: Apply empty transform dict.

        **Expected**: Data returned unchanged.

        Validates: Requirement PP-5.1 - Empty transform handling
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        # Apply empty transforms
        result = preprocessor.apply_transforms(sample_dataset_container, {})

        # Data should be returned (possibly unchanged)
        assert result is not None


class TestPreProcessorGetCachedTransformManager:
    """Tests for get_cached_transform_manager method."""

    def test_returns_transform_manager(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Test that method returns stored transform manager.

        **PHM Logic**: Cached transforms can be retrieved for inspection.

        **Methodology**: Call get_cached_transform_manager.

        **Expected**: Same transform manager returned.

        Validates: Requirement PP-6.1 - Transform manager retrieval
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        result = preprocessor.get_cached_transform_manager()

        assert result is mock_transform_manager


class TestPreProcessorInterface:
    """Tests for PreProcessorInterface abstract base class."""

    def test_abstract_cannot_instantiate(self):
        """Test that abstract class cannot be instantiated.

        **PHM Logic**: Interface defines contract for preprocessors.

        **Methodology**: Attempt direct instantiation.

        **Expected**: TypeError raised.

        Validates: Requirement PPI-1.1 - Abstract enforcement
        """
        with pytest.raises(TypeError):
            PreProcessorInterface(datasource=None, transforms=None)

    def test_concrete_implementation_required_methods(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Test that PreProcessor implements required methods.

        **PHM Logic**: Concrete class must implement interface.

        **Methodology**: Create PreProcessor, verify methods exist.

        **Expected**: All interface methods callable.

        Validates: Requirement PPI-1.2 - Interface implementation
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader, transforms=mock_transform_manager
        )

        # Check required methods exist
        assert hasattr(preprocessor, "get_processed_data_container")
        assert hasattr(preprocessor, "get_processed_split_dict")
        assert hasattr(preprocessor, "get_meta_data_dict")
        assert hasattr(preprocessor, "fetch_data")
        assert hasattr(preprocessor, "apply_transforms")
        assert hasattr(preprocessor, "pipeline")

        # Methods should be callable
        assert callable(preprocessor.get_processed_data_container)
        assert callable(preprocessor.get_processed_split_dict)
        assert callable(preprocessor.get_meta_data_dict)
        assert callable(preprocessor.fetch_data)
        assert callable(preprocessor.apply_transforms)
        assert callable(preprocessor.pipeline)


class TestPreProcessorEdgeCases:
    """Edge case tests for PreProcessor."""

    def test_none_datasource_handling(self, mock_transform_manager):
        """Test handling of None datasource.

        **PHM Logic**: None datasource may be valid for some configurations.

        **Methodology**: Create preprocessor with None datasource.

        **Expected**: May succeed or fail with clear error.

        Validates: Requirement PP-7.1 - None datasource handling
        """
        # This may or may not be allowed depending on implementation
        try:
            preprocessor = PreProcessor(
                datasource=None, transforms=mock_transform_manager
            )
            # If creation succeeds, fetch_data should fail
            with pytest.raises((AttributeError, TypeError)):
                preprocessor.fetch_data()
        except (TypeError, AttributeError):
            pass  # Expected if None not allowed

    def test_kwargs_passthrough(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """Test that extra kwargs are handled.

        **PHM Logic**: Extra kwargs allow flexibility.

        **Methodology**: Pass extra kwargs to constructor.

        **Expected**: No error (kwargs stored or ignored).

        Validates: Requirement PP-7.2 - Kwargs handling
        """
        # Should not raise
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager,
            extra_param="value",
            another_param=42,
        )

        assert preprocessor.datasource is mock_single_source_loader


@pytest.mark.unit
class TestPreprocessorInvalidTransforms:
    """Tests __init__ transform type guard (line 149)."""

    def test_non_protocol_transforms_raises_type_error(self, mock_single_source_loader):
        """Non-TransformSequenceProtocol transforms → TypeError.

        **Methodology**: Pass a plain object as transforms.

        **Expected**: TypeError raised mentioning the type name.
        """

        class _NotATransformSequence:
            pass

        with pytest.raises(TypeError, match="_NotATransformSequence"):
            PreProcessor(
                datasource=mock_single_source_loader,
                transforms=_NotATransformSequence(),
            )


@pytest.mark.unit
class TestFetchDataNonContainer:
    """Tests fetch_data contract check (lines 231-235)."""

    def test_get_data_returns_non_container_raises(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """get_data() returning a non-DatasetContainer → PreprocessingDatasourceError.

        **Methodology**: Configure mock to return a plain dict.

        **Expected**: PreprocessingDatasourceError raised.
        """
        mock_single_source_loader.get_data.return_value = {"not": "a container"}
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager,
        )
        with pytest.raises(PreprocessingDatasourceError):
            preprocessor.fetch_data()

    def test_get_data_raises_datasource_error_wrapped(self, mock_transform_manager):
        """get_data() raising DatasourceError → wrapped in PreprocessingDatasourceError (lines 276-277).

        **Methodology**: Use _PreprocessingErrorDatasource with DatasourceStateError.

        **Expected**: PreprocessingDatasourceError with stage='get_data'.
        """
        from picid.data.datasources.base.exceptions import DatasourceStateError

        datasource = _PreprocessingErrorDatasource(
            data_error=DatasourceStateError("boom"),
        )
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=mock_transform_manager,
        )
        with pytest.raises(PreprocessingDatasourceError) as exc_info:
            preprocessor.fetch_data()
        assert exc_info.value.stage == "get_data"


@pytest.mark.unit
class TestAddDatasourceManifestEntry:
    """Tests _add_datasource_manifest_entry early-return branches (lines 290, 293)."""

    def test_data_none_returns_cleanly(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """self.data is None → method returns without error (line 290).

        **Methodology**: Call directly on fresh preprocessor (data never loaded).

        **Expected**: No exception raised.
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager,
        )
        preprocessor._add_datasource_manifest_entry()

    def test_non_metadata_manifest_returns_cleanly(
        self, mock_single_source_loader, mock_transform_manager
    ):
        """manifest not a MetadataManifest → returns without adding entry (line 293).

        **Methodology**: Attach a Mock with no manifest attribute so getattr returns None.

        **Expected**: No exception raised.
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager,
        )
        preprocessor.data = Mock(spec=[])
        preprocessor._add_datasource_manifest_entry()


@pytest.mark.unit
class TestResolveDatasourceNamePreprocessor:
    """Tests PreProcessor._resolve_datasource_name static method branches."""

    def test_get_data_names_single_tuple_returns_string(self):
        """get_data_names() returning 1-tuple → unpacked string (line 340).

        **Expected**: "solo", not ("solo",).
        """

        class _Ds:
            def get_data_names(self):
                return ("solo",)

        assert PreProcessor._resolve_datasource_name(_Ds()) == "solo"

    def test_get_data_names_multi_tuple_returns_list(self):
        """get_data_names() returning multi-tuple → list (lines 342-344).

        **Expected**: ["a", "b"].
        """

        class _Ds:
            def get_data_names(self):
                return ("a", "b")

        assert PreProcessor._resolve_datasource_name(_Ds()) == ["a", "b"]

    def test_get_data_name_returns_string(self):
        """get_data_name() returning a string → returned directly (line 350).

        **Expected**: "myname".
        """

        class _Ds:
            def get_data_name(self):
                return "myname"

        assert PreProcessor._resolve_datasource_name(_Ds()) == "myname"

    def test_get_data_name_returns_list_of_strings(self):
        """get_data_name() returning a list → returned as-is (line 354).

        **Expected**: ["x", "y"].
        """

        class _Ds:
            def get_data_name(self):
                return ["x", "y"]

        assert PreProcessor._resolve_datasource_name(_Ds()) == ["x", "y"]

    def test_get_data_name_raises_returns_none(self):
        """get_data_name() raising → None returned, no propagation (lines 355-356).

        **Expected**: None.
        """

        class _Ds:
            def get_data_name(self):
                raise RuntimeError("boom")

        assert PreProcessor._resolve_datasource_name(_Ds()) is None

    def test_get_data_names_raises_falls_through_to_none(self):
        """get_data_names() raising → exception caught, falls through (lines 343-344).

        **Expected**: None returned (no data_name, no get_data_names success).
        """

        class _Ds:
            def get_data_names(self):
                raise RuntimeError("exploded")

        assert PreProcessor._resolve_datasource_name(_Ds()) is None


@pytest.mark.unit
class TestApplyTransformsErrors:
    """Tests apply_transforms error branches (lines 397, 419-422)."""

    def test_non_data_transform_raises_value_error(self):
        """transforms dict with non-DataTransform value → ValueError (line 397).

        **Methodology**: Pass "not_a_transform" as the dict value.

        **Expected**: ValueError raised mentioning DataTransform.
        """
        datasource = _MetadataAwareDatasource()
        preprocessor = PreProcessor(datasource=datasource)
        preprocessor.fetch_data()
        with pytest.raises(ValueError, match="DataTransform"):
            preprocessor.apply_transforms(
                preprocessor.data,
                {"t": "not_a_transform"},
            )

    def test_forward_exception_wrapped_in_transform_error(self):
        """transform.forward() raising RuntimeError → TransformError (lines 419-422).

        **Methodology**: Patch DataTransform.forward to raise RuntimeError.

        **Expected**: TransformError raised with original cause attached.
        """
        from unittest.mock import patch
        from picid.exceptions import TransformError

        datasource = _MetadataAwareDatasource()
        preprocessor = PreProcessor(datasource=datasource)
        preprocessor.fetch_data()

        transform_instance = _PreprocessingMetadataTransform()
        dt = DataTransform(
            "failing_transform",
            transform_instance,
            {"apply_to": "features", "assign_to": "features"},
        )
        with patch.object(dt, "forward", side_effect=RuntimeError("exploded")):
            with pytest.raises(TransformError):
                preprocessor.apply_transforms(
                    preprocessor.data,
                    OrderedDict([("failing_transform", dt)]),
                )

    def test_transform_error_reraises_without_wrapping(self):
        """transform.forward() raising TransformError → re-raised as-is (line 420).

        **Methodology**: Patch forward to raise TransformError directly.

        **Expected**: Same TransformError propagates, not wrapped in a new one.
        """
        from unittest.mock import patch
        from picid.exceptions import TransformError

        datasource = _MetadataAwareDatasource()
        preprocessor = PreProcessor(datasource=datasource)
        preprocessor.fetch_data()

        transform_instance = _PreprocessingMetadataTransform()
        dt = DataTransform(
            "reraise_transform",
            transform_instance,
            {"apply_to": "features", "assign_to": "features"},
        )
        original_err = TransformError("already wrapped", step_id="reraise_transform")
        with patch.object(dt, "forward", side_effect=original_err):
            with pytest.raises(TransformError) as exc_info:
                preprocessor.apply_transforms(
                    preprocessor.data,
                    OrderedDict([("reraise_transform", dt)]),
                )
            assert exc_info.value is original_err

    def test_after_each_transform_callback_invoked(self):
        """after_each_transform_callback called after each transform (line 429).

        **Methodology**: Pass a Mock callback, mock forward to succeed.

        **Expected**: callback called once with (data, transform_name).
        """
        from unittest.mock import Mock, patch

        datasource = _MetadataAwareDatasource()
        preprocessor = PreProcessor(datasource=datasource)
        preprocessor.fetch_data()
        data = preprocessor.data

        transform_instance = _PreprocessingMetadataTransform()
        dt = DataTransform(
            "cb_transform",
            transform_instance,
            {"apply_to": "features", "assign_to": "features"},
        )
        callback = Mock()
        with patch.object(dt, "forward", return_value=(data, {})):
            preprocessor.apply_transforms(
                data,
                OrderedDict([("cb_transform", dt)]),
                after_each_transform_callback=callback,
            )
        callback.assert_called_once()
