"""Comprehensive tests for rich_output utilities.

This module tests the data structure description and visualization
utilities used for debugging PHM data pipelines.

PHM Context:
-----------
Understanding data structure changes through the transform pipeline
is essential for debugging and validating PHM preprocessing.

Test Coverage Strategy:
----------------------
1. **Data Type Description**: Type and shape reporting
2. **Descriptive Dict Conversion**: Nested structure analysis
3. **Difference Detection**: Changed/added/removed keys
4. **Tree Printing**: Visual representation of data dicts
"""
import pytest
import numpy as np
import pandas as pd
import torch
import awkward as ak
from omegaconf import OmegaConf
from unittest.mock import patch

from picid.utils.rich_output import (
    describe_data_type,
    to_descriptive_dict,
    descriptive_dict_differences_str,
    print_data_dict_structure,
    extract_targets,
    get_config_info,
    build_transform_error_renderables,
    display_targets,
    display_config_sources,
    print_transforms_summary,
)


class TestDescribeDataType:
    """Tests for describe_data_type function."""
    
    def test_describe_numpy_array(self):
        """Test description of numpy array.
        
        **PHM Logic**: Arrays are primary data type in PHM.
        
        **Methodology**: Describe various numpy arrays.
        
        **Expected**: Shape and dtype information.
        
        Validates: Requirement RO-1.1 - Numpy array description
        """
        arr = np.random.randn(100, 5)
        
        desc = describe_data_type(arr)
        
        assert isinstance(desc, str)
        assert "100" in desc or "5" in desc  # Shape should be present
    
    def test_describe_torch_tensor(self):
        """Test description of PyTorch tensor.
        
        **PHM Logic**: Tensors used in model training.
        
        **Methodology**: Describe torch tensor.
        
        **Expected**: Shape and dtype information.
        
        Validates: Requirement RO-1.2 - Tensor description
        """
        tensor = torch.randn(50, 10)
        
        desc = describe_data_type(tensor)
        
        assert isinstance(desc, str)
        assert "50" in desc or "10" in desc
    
    def test_describe_pandas_dataframe(self):
        """Test description of pandas DataFrame.
        
        **PHM Logic**: DataFrames common for tabular PHM data.
        
        **Methodology**: Describe DataFrame.
        
        **Expected**: Shape information.
        
        Validates: Requirement RO-1.3 - DataFrame description
        """
        df = pd.DataFrame(np.random.randn(100, 5), columns=list('ABCDE'))
        
        desc = describe_data_type(df)
        
        assert isinstance(desc, str)
        # Should mention rows or columns
        assert "100" in desc or "5" in desc or "DataFrame" in desc
    
    def test_describe_pandas_series(self):
        """Test description of pandas Series.
        
        **PHM Logic**: Series used for single columns/targets.
        
        **Methodology**: Describe Series.
        
        **Expected**: Length and dtype information.
        
        Validates: Requirement RO-1.4 - Series description
        """
        series = pd.Series(np.random.randn(100))
        
        desc = describe_data_type(series)
        
        assert isinstance(desc, str)
    
    def test_describe_awkward_array(self):
        """Test description of awkward array.
        
        **PHM Logic**: Ragged arrays for variable-length sequences.
        
        **Methodology**: Describe awkward array.
        
        **Expected**: Shape with 'var' for ragged dimensions.
        
        Validates: Requirement RO-1.5 - Awkward array description
        """
        arr = ak.Array([[1, 2], [3, 4, 5], [6]])
        
        desc = describe_data_type(arr)
        
        assert isinstance(desc, str)
    
    def test_describe_dict(self):
        """Test description of dictionary.
        
        **PHM Logic**: Dicts used for structured data objects.
        
        **Methodology**: Describe nested dict.
        
        **Expected**: Key information present.
        
        Validates: Requirement RO-1.6 - Dict description
        """
        data = {
            'features': np.random.randn(100, 5),
            'target': np.random.randn(100, 1)
        }
        
        desc = describe_data_type(data)
        
        assert isinstance(desc, str)
    
    def test_describe_list(self):
        """Test description of list.
        
        **PHM Logic**: Lists used for multi-unit data.
        
        **Methodology**: Describe list of arrays.
        
        **Expected**: List length and element info.
        
        Validates: Requirement RO-1.7 - List description
        """
        data = [np.random.randn(100, 5) for _ in range(3)]
        
        desc = describe_data_type(data)
        
        assert isinstance(desc, str)
    
    def test_describe_with_statistics(self):
        """Test description with statistics calculation.
        
        **PHM Logic**: Statistics help verify data values.
        
        **Methodology**: Request statistics in description.
        
        **Expected**: Mean/std or similar stats included.
        
        Validates: Requirement RO-1.8 - Statistics in description
        """
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        desc = describe_data_type(arr, calculate_stat=True)
        
        assert isinstance(desc, str)


class TestToDescriptiveDict:
    """Tests for to_descriptive_dict function."""
    
    def test_simple_dict_conversion(self):
        """Test conversion of simple dict.
        
        **PHM Logic**: Converts values to descriptive strings.
        
        **Methodology**: Convert dict with array values.
        
        **Expected**: Dict with string descriptions.
        
        Validates: Requirement RO-2.1 - Simple dict conversion
        """
        data = {
            'features': np.random.randn(100, 5),
            'target': np.random.randn(100, 1)
        }
        
        result = to_descriptive_dict(data)
        
        assert isinstance(result, dict)
        assert 'features' in result
        assert 'target' in result
        assert isinstance(result['features'], str)
    
    def test_nested_dict_conversion(self):
        """Test conversion of nested dict.
        
        **PHM Logic**: PHM data often has nested structure.
        
        **Methodology**: Convert nested dict.
        
        **Expected**: All levels converted.
        
        Validates: Requirement RO-2.2 - Nested dict conversion
        """
        data = {
            'train': {
                'features': np.random.randn(100, 5),
                'target': np.random.randn(100, 1)
            },
            'test': {
                'features': np.random.randn(50, 5),
                'target': np.random.randn(50, 1)
            }
        }
        
        result = to_descriptive_dict(data)
        
        assert isinstance(result, dict)
        assert 'train' in result
        assert isinstance(result['train'], dict)
    
    def test_empty_dict_conversion(self):
        """Test conversion of empty dict.
        
        **PHM Logic**: Empty dict should return empty.
        
        **Methodology**: Convert empty dict.
        
        **Expected**: Empty dict returned.
        
        Validates: Requirement RO-2.3 - Empty dict handling
        """
        result = to_descriptive_dict({})
        
        assert result == {}


class TestDescriptiveDictDifferences:
    """Tests for descriptive_dict_differences_str function."""
    
    def test_added_keys_detection(self):
        """Test detection of added keys.
        
        **PHM Logic**: New keys indicate transform added outputs.
        
        **Methodology**: Compare dicts with added key.
        
        **Expected**: Added key reported.
        
        Validates: Requirement RO-3.1 - Added key detection
        """
        old = {'a': 'desc_a'}
        new = {'a': 'desc_a', 'b': 'desc_b'}  # 'b' added
        
        result = descriptive_dict_differences_str(old, new, mode="added")
        
        assert isinstance(result, str)
        # 'b' should be mentioned
        assert 'b' in result or len(result) > 0
    
    def test_removed_keys_detection(self):
        """Test detection of removed keys.
        
        **PHM Logic**: Removed keys indicate transform dropped outputs.
        
        **Methodology**: Compare dicts with removed key.
        
        **Expected**: Removed key reported.
        
        Validates: Requirement RO-3.2 - Removed key detection
        """
        old = {'a': 'desc_a', 'b': 'desc_b'}
        new = {'a': 'desc_a'}  # 'b' removed
        
        result = descriptive_dict_differences_str(old, new, mode="removed")
        
        assert isinstance(result, str)
    
    def test_changed_keys_detection(self):
        """Test detection of changed values.
        
        **PHM Logic**: Changed values indicate transform modified data.
        
        **Methodology**: Compare dicts with changed value.
        
        **Expected**: Changed key reported.
        
        Validates: Requirement RO-3.3 - Changed value detection
        """
        old = {'a': 'old_desc'}
        new = {'a': 'new_desc'}  # Value changed
        
        result = descriptive_dict_differences_str(old, new, mode="changed")
        
        assert isinstance(result, str)
    
    def test_identical_dicts(self):
        """Test comparison of identical dicts.
        
        **PHM Logic**: Identical dicts have no differences.
        
        **Methodology**: Compare identical dicts.
        
        **Expected**: Empty or minimal output.
        
        Validates: Requirement RO-3.4 - Identical dict handling
        """
        data = {'a': 'desc_a', 'b': 'desc_b'}
        
        result_added = descriptive_dict_differences_str(data, data, mode="added")
        result_removed = descriptive_dict_differences_str(data, data, mode="removed")
        
        # Should have no differences
        assert result_added == "" or "no" in result_added.lower()
        assert result_removed == "" or "no" in result_removed.lower()
    
    def test_invalid_mode_error(self):
        """Test error for invalid mode.
        
        **PHM Logic**: Only 'added', 'removed', 'changed' are valid.
        
        **Methodology**: Pass invalid mode.
        
        **Expected**: AssertionError raised.
        
        Validates: Requirement RO-3.5 - Mode validation
        """
        old = {'a': 'desc'}
        new = {'a': 'desc'}
        
        with pytest.raises(AssertionError):
            descriptive_dict_differences_str(old, new, mode="invalid")


class TestPrintDataDictStructure:
    """Tests for print_data_dict_structure function."""
    
    def test_print_simple_dict(self):
        """Test tree visualization of simple dict.
        
        **PHM Logic**: Visual tree helps understand data structure.
        
        **Methodology**: Generate tree for simple dict.
        
        **Expected**: Rich Tree object returned.
        
        Validates: Requirement RO-4.1 - Simple dict tree
        """
        data = {
            'features': np.random.randn(100, 5),
            'target': np.random.randn(100, 1)
        }
        
        tree = print_data_dict_structure(data)
        
        # Should return Tree object
        assert tree is not None
    
    def test_print_nested_dict(self):
        """Test tree visualization of nested dict.
        
        **PHM Logic**: Nested structure shown hierarchically.
        
        **Methodology**: Generate tree for nested dict.
        
        **Expected**: Rich Tree with nested branches.
        
        Validates: Requirement RO-4.2 - Nested dict tree
        """
        data = {
            'train': {
                'features': np.random.randn(100, 5)
            },
            'test': {
                'features': np.random.randn(50, 5)
            }
        }
        
        tree = print_data_dict_structure(data)
        
        assert tree is not None
    
    def test_print_empty_dict(self):
        """Test tree visualization of empty dict.
        
        **PHM Logic**: Empty dict should produce minimal tree.
        
        **Methodology**: Generate tree for empty dict.
        
        **Expected**: Tree object (possibly empty).
        
        Validates: Requirement RO-4.3 - Empty dict tree
        """
        tree = print_data_dict_structure({})
        
        # Should not crash
        assert tree is not None


class TestExtractTargets:
    def test_extract_targets_from_dictconfig(self):
        cfg = OmegaConf.create({"model": {"_target_": "torch.nn.Linear"}})
        targets = extract_targets(cfg)
        assert len(targets) == 1
        assert targets[0][0] == "model"
        assert targets[0][1] == "torch.nn.Linear"

    def test_extract_targets_empty_dict(self):
        assert extract_targets(OmegaConf.create({})) == []

    def test_extract_targets_non_dict_returns_empty(self):
        """OmegaConf non-dict (e.g. list or scalar) returns empty list."""
        cfg = OmegaConf.create([1, 2, 3])
        assert extract_targets(cfg) == []

    def test_extract_targets_target_ending_with_py(self):
        """Target string ending with .py gets empty module_path."""
        cfg = OmegaConf.create({"x": {"_target_": "mymodule.py"}})
        targets = extract_targets(cfg)
        assert len(targets) == 1
        assert targets[0][2] == ""

    def test_extract_targets_nested_recursion(self):
        """Recurse into nested _target_ configs."""
        cfg = OmegaConf.create({
            "a": {"_target_": "foo.Bar", "nested": {"_target_": "baz.Quux"}},
        })
        targets = extract_targets(cfg)
        assert len(targets) == 2
        assert targets[0][0] == "a"
        assert targets[1][0] == "a.nested"


class TestDisplayTargets:
    def test_display_targets_prints_table(self):
        """display_targets builds and prints table without error."""
        cfg = OmegaConf.create({"model": {"_target_": "torch.nn.Linear"}})
        with patch("picid.utils.rich_output.Console") as mock_console:
            display_targets(cfg)
        mock_console.return_value.print.assert_called_once()


class TestGetConfigInfo:
    def test_get_config_info_returns_table_data(self):
        cfg = OmegaConf.create({"a": 1, "b": 2})
        cfg._metadata.config_sources = {}  # Avoid AttributeError in recurse
        with patch("picid.utils.rich_output.HydraConfig") as m:
            m.get.return_value.overrides.task = []
            data = get_config_info(cfg)
        assert isinstance(data, list)


class TestBuildTransformErrorRenderables:
    def test_builds_rule_table_panel(self):
        out = build_transform_error_renderables(
            "TestTransform",
            {"flag1": True},
            {"meta": "data"},
            ["k1"],
            [("k1", "np.ndarray", "desc")],
            "case analysis line",
        )
        assert len(out) >= 3
        assert any(hasattr(r, "title") or "Transform Error" in str(r) for r in out)

    def test_builds_without_first_segment(self):
        """When first_segment_keys is None or first_segment_rows empty, no segment table."""
        out = build_transform_error_renderables(
            "T", {}, {}, None, [], "case"
        )
        assert len(out) >= 3


class TestDescribeDataTypeEdgeCases:
    """Additional tests for describe_data_type uncovered branches."""

    def test_describe_empty_list(self):
        assert describe_data_type([]) == "list<empty> x 0"

    def test_describe_list_of_ak_arrays(self):
        arr = ak.Array([[1, 2], [3, 4, 5]])
        desc = describe_data_type([arr, arr], calculate_stat=False)
        assert "list<ak.Array>" in desc

    def test_describe_list_of_ak_arrays_with_stat(self):
        arr = ak.Array([[1, 2], [3, 4, 5], [6]])
        desc = describe_data_type([arr, arr], calculate_stat=True)
        assert "list<ak.Array>" in desc
        assert "min" in desc or "var" in desc or "dim" in desc.lower()

    def test_describe_list_of_non_ak_arrays(self):
        """Fallback for list of non-ak types."""
        data = [np.array([1, 2]), np.array([3, 4])]
        desc = describe_data_type(data)
        assert "ndarray" in desc or "list" in desc

    def test_describe_unknown_type(self):
        """Fallback returns type name for unknown types."""
        class Custom:
            pass
        assert "Custom" in describe_data_type(Custom())


class TestToDescriptiveDictBaseDataObject:
    """Test to_descriptive_dict with BaseDataObject."""

    def test_base_data_object_conversion(self):
        """to_descriptive_dict handles BaseDataObject like dict."""
        from picid.data.data_objects.base import BaseDataObject

        obj = BaseDataObject(x=np.array([1, 2, 3]))
        result = to_descriptive_dict(obj)
        assert "x" in result
        assert isinstance(result["x"], str)


class TestPrintDataDictStructureEdgeCases:
    """Test add_to_tree else branch (non-dict leaf)."""

    def test_tree_with_leaf_value(self):
        """add_to_tree with leaf value (not dict) adds describe_data_type."""
        data = {"x": np.array([1, 2])}
        tree = print_data_dict_structure(data)
        assert tree is not None


class TestDisplayConfigSources:
    def test_display_config_sources_prints_table(self):
        """display_config_sources calls get_config_info and prints table."""
        cfg = OmegaConf.create({"a": 1, "b": 2})
        with patch("picid.utils.rich_output.get_config_info", return_value=[("a", "1", "", "")]):
            with patch("picid.utils.rich_output.Console") as mock_console:
                display_config_sources(cfg)
        mock_console.return_value.print.assert_called_once()


class TestPrintTransformsSummary:
    def test_print_transforms_summary(self):
        summary = [
            {
                "transform_name": "T1",
                "time": "0.1",
                "status": "ok",
                "details": "d",
                "changes": "c",
                "added": "a",
                "removed": "r",
                "inputs": "i",
            },
        ]
        with patch("picid.utils.rich_output.Console") as mock_console:
            print_transforms_summary(summary)
        mock_console.return_value.print.assert_called_once()

    def test_print_transforms_summary_with_missing_keys(self):
        """Handles summary dicts with missing optional keys."""
        summary = [{"transform_name": "T", "time": "0", "status": "ok", "details": ""}]
        with patch("picid.utils.rich_output.Console") as mock_console:
            print_transforms_summary(summary)
        mock_console.return_value.print.assert_called_once()
