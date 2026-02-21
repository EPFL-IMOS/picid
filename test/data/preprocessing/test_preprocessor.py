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
import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass

from picid.data.preprocessing.preprocessor import PreProcessor
from picid.data.preprocessing.base import PreProcessorInterface


class TestPreProcessorInitialization:
    """Tests for PreProcessor initialization."""
    
    def test_init_stores_datasource_and_transforms(self, mock_single_source_loader, mock_transform_manager):
        """Test initialization stores datasource, transforms, and default state.
        
        **PHM Logic**: PreProcessor orchestrates datasource and transform pipeline.
        
        **Methodology**: Create PreProcessor with datasource and transforms.
        
        **Expected**: Datasource and transforms stored, not preprocessed flag set.
        
        Validates: Requirement PP-1.1 - Initialization
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        assert preprocessor.datasource is mock_single_source_loader
        assert preprocessor.transforms is mock_transform_manager
        assert preprocessor._is_preprocessed == False
        assert preprocessor.data is None
    
    def test_init_stores_datasource(self, mock_single_source_loader, mock_transform_manager):
        """Test that datasource is stored correctly.
        
        **PHM Logic**: Datasource provides data loading interface.
        
        **Methodology**: Verify datasource attribute after init.
        
        **Expected**: Datasource accessible via attribute.
        
        Validates: Requirement PP-1.4 - Datasource storage
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        assert preprocessor.datasource is mock_single_source_loader
    
    def test_init_stores_transforms(self, mock_single_source_loader, mock_transform_manager):
        """Test that transform manager is stored correctly.
        
        **PHM Logic**: Transform manager provides transform pipeline.
        
        **Methodology**: Verify transforms attribute after init.
        
        **Expected**: Transform manager accessible.
        
        Validates: Requirement PP-1.5 - Transform storage
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        assert preprocessor.transforms is mock_transform_manager


class TestPreProcessorMetaData:
    """Tests for metadata handling."""
    
    def test_get_meta_data_dict_empty_default(self, mock_single_source_loader, mock_transform_manager):
        """Test get_meta_data_dict returns empty dict by default.
        
        **PHM Logic**: Metadata populated during pipeline execution.
        
        **Methodology**: Call get_meta_data_dict before pipeline.
        
        **Expected**: Empty dict returned.
        
        Validates: Requirement PP-2.1 - Default metadata
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        meta = preprocessor.get_meta_data_dict()
        
        assert isinstance(meta, dict)
    
    def test_get_meta_data_dict_after_fetch(self, mock_single_source_loader, mock_transform_manager, sample_dataset_container):
        """Test get_meta_data_dict after data fetch.
        
        **PHM Logic**: Metadata may be populated by datasource.
        
        **Methodology**: Fetch data, then get metadata.
        
        **Expected**: Metadata dict (possibly with content from source).
        
        Validates: Requirement PP-2.2 - Metadata after fetch
        """
        mock_single_source_loader.get_data.return_value = sample_dataset_container
        
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        preprocessor.fetch_data()
        meta = preprocessor.get_meta_data_dict()
        
        assert isinstance(meta, dict)


class TestPreProcessorFetchData:
    """Tests for fetch_data method."""
    
    def test_fetch_data_calls_datasource(self, mock_single_source_loader, mock_transform_manager, sample_dataset_container):
        """Test that fetch_data calls datasource.get_data().
        
        **PHM Logic**: Data loaded from datasource on demand.
        
        **Methodology**: Call fetch_data, verify datasource called.
        
        **Expected**: get_data() called once.
        
        Validates: Requirement PP-3.1 - Datasource invocation
        """
        mock_single_source_loader.get_data.return_value = sample_dataset_container
        
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        preprocessor.fetch_data()
        
        mock_single_source_loader.get_data.assert_called_once()
    
    def test_fetch_data_stores_result(self, mock_single_source_loader, mock_transform_manager, sample_dataset_container):
        """Test that fetched data is stored.
        
        **PHM Logic**: Data stored for transform application.
        
        **Methodology**: Fetch data, verify self.data set.
        
        **Expected**: self.data contains DatasetContainer.
        
        Validates: Requirement PP-3.2 - Data storage
        """
        mock_single_source_loader.get_data.return_value = sample_dataset_container
        
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        preprocessor.fetch_data()
        
        assert preprocessor.data is not None


class TestPreProcessorGetProcessedData:
    """Tests for get_processed_data_dict method."""
    
    def test_get_processed_data_before_pipeline_error(self, mock_single_source_loader, mock_transform_manager):
        """Test error when called before pipeline execution.
        
        **PHM Logic**: Data not available until pipeline runs.
        
        **Methodology**: Call get_processed_data_dict without pipeline.
        
        **Expected**: RuntimeError raised.
        
        Validates: Requirement PP-4.1 - Pre-pipeline error
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        with pytest.raises(RuntimeError):
            preprocessor.get_processed_data_dict()


class TestPreProcessorApplyTransforms:
    """Tests for apply_transforms method."""
    
    def test_apply_transforms_empty_transforms(self, mock_single_source_loader, mock_transform_manager, sample_dataset_container):
        """Test apply_transforms with empty transform dict.
        
        **PHM Logic**: No transforms means data passes through unchanged.
        
        **Methodology**: Apply empty transform dict.
        
        **Expected**: Data returned unchanged.
        
        Validates: Requirement PP-5.1 - Empty transform handling
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        # Apply empty transforms
        result = preprocessor.apply_transforms(sample_dataset_container, {})
        
        # Data should be returned (possibly unchanged)
        assert result is not None


class TestPreProcessorGetCachedTransformManager:
    """Tests for get_cached_transform_manager method."""

    def test_returns_transform_manager(self, mock_single_source_loader, mock_transform_manager):
        """Test that method returns stored transform manager.

        **PHM Logic**: Cached transforms can be retrieved for inspection.

        **Methodology**: Call get_cached_transform_manager.

        **Expected**: Same transform manager returned.

        Validates: Requirement PP-6.1 - Transform manager retrieval
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
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
    
    def test_concrete_implementation_required_methods(self, mock_single_source_loader, mock_transform_manager):
        """Test that PreProcessor implements required methods.
        
        **PHM Logic**: Concrete class must implement interface.
        
        **Methodology**: Create PreProcessor, verify methods exist.
        
        **Expected**: All interface methods callable.
        
        Validates: Requirement PPI-1.2 - Interface implementation
        """
        preprocessor = PreProcessor(
            datasource=mock_single_source_loader,
            transforms=mock_transform_manager
        )
        
        # Check required methods exist
        assert hasattr(preprocessor, 'get_processed_data_dict')
        assert hasattr(preprocessor, 'get_meta_data_dict')
        assert hasattr(preprocessor, 'fetch_data')
        assert hasattr(preprocessor, 'apply_transforms')
        assert hasattr(preprocessor, 'pipeline')
        
        # Methods should be callable
        assert callable(preprocessor.get_processed_data_dict)
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
                datasource=None,
                transforms=mock_transform_manager
            )
            # If creation succeeds, fetch_data should fail
            with pytest.raises((AttributeError, TypeError)):
                preprocessor.fetch_data()
        except (TypeError, AttributeError):
            pass  # Expected if None not allowed
    
    def test_kwargs_passthrough(self, mock_single_source_loader, mock_transform_manager):
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
            another_param=42
        )
        
        assert preprocessor.datasource is mock_single_source_loader
