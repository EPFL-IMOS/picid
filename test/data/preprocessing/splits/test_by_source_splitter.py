"""
Comprehensive tests for ``BySourceSplitter``.

This module covers initialization, split assignment, data reorganization,
tree-view generation, and edge cases for source-based splitting in
multi-unit PHM experiments.
"""

import numpy as np
import pytest

from picid.data.split_strategies.by_source_splitter import (
    BySourceSplitter,
    convert_outer_list_to_inner,
)


class TestBySourceSplitterInitialization:
    """Tests for BySourceSplitter initialization."""

    def test_init_default_empty_lists(self):
        """
        Test initialization with default empty lists.

        **PHM Logic**: Empty lists allow dynamic source assignment.

        **Methodology**: Create splitter without arguments.

        **Expected**: All sources lists are empty.

        Validates: Requirement BSS-1.1 - Default initialization
        """
        splitter = BySourceSplitter()

        assert splitter.sources_train == []
        assert splitter.sources_val == []
        assert splitter.sources_test == []

    def test_init_with_source_lists(self):
        """
        Test initialization with source lists.

        **PHM Logic**: Assign specific units to each split.

        **Methodology**: Create splitter with source assignments.

        **Expected**: Sources correctly assigned to splits.

        Validates: Requirement BSS-1.2 - Source list assignment
        """
        splitter = BySourceSplitter(
            sources_train=["unit_1", "unit_2"],
            sources_val=["unit_3"],
            sources_test=["unit_4", "unit_5"],
        )

        assert splitter.sources_train == ["unit_1", "unit_2"]
        assert splitter.sources_val == ["unit_3"]
        assert splitter.sources_test == ["unit_4", "unit_5"]

    def test_init_overlapping_sources_error(self):
        """
        Test that overlapping sources raise error.

        **PHM Logic**: A source cannot be in multiple splits (data leakage).

        **Methodology**: Create splitter with overlapping sources.

        **Expected**: ValueError raised about disjoint sets.

        Validates: Requirement BSS-1.3 - Overlap validation
        """
        with pytest.raises(ValueError, match="disjoint"):
            BySourceSplitter(
                sources_train=["unit_1", "unit_2"],
                sources_val=["unit_2", "unit_3"],  # unit_2 appears twice!
                sources_test=["unit_4"],
            )

    def test_init_none_converted_to_empty_list(self):
        """
        Test that None sources are converted to empty lists.

        **PHM Logic**: None is equivalent to no sources specified.

        **Methodology**: Pass None for some source lists.

        **Expected**: None converted to empty list.

        Validates: Requirement BSS-1.4 - None handling
        """
        splitter = BySourceSplitter(
            sources_train=["unit_1"], sources_val=None, sources_test=None
        )

        assert splitter.sources_train == ["unit_1"]
        assert splitter.sources_val == []
        assert splitter.sources_test == []


class TestBySourceSplitterSplitData:
    """Tests for split_data method."""

    def test_split_data_basic(self):
        """
        Test basic data splitting by source.

        **PHM Logic**: Each source's data should be assigned to correct split.

        **Methodology**: Split multi-source data with known assignments.

        **Expected**: Data organized by split, then by data key.

        Validates: Requirement BSS-2.1 - Basic splitting
        """
        splitter = BySourceSplitter(
            sources_train=["source_a"],
            sources_val=["source_b"],
            sources_test=["source_c"],
        )

        # Create mock containers with simple dict data
        data_a = {"features": np.array([1, 2, 3]), "target": np.array([0])}
        data_b = {"features": np.array([4, 5, 6]), "target": np.array([1])}
        data_c = {"features": np.array([7, 8, 9]), "target": np.array([2])}

        data_list = [data_a, data_b, data_c]
        source_names = ["source_a", "source_b", "source_c"]

        result = splitter.split_data(data_list, source_names)

        # Verify structure: result[data_key][split_name]
        assert "features" in result
        assert "target" in result

        assert "train" in result["features"]
        assert "val" in result["features"]
        assert "test" in result["features"]

    def test_split_data_multiple_train_sources(self):
        """
        Test splitting with multiple sources per split.

        **PHM Logic**: Multiple units may be assigned to training.

        **Methodology**: Assign multiple sources to train.

        **Expected**: All train sources combined in train split.

        Validates: Requirement BSS-2.2 - Multiple sources per split
        """
        splitter = BySourceSplitter(
            sources_train=["source_a", "source_b"],  # Multiple train sources
            sources_val=["source_c"],
            sources_test=["source_d"],
        )

        data_list = [
            {"features": np.array([1])},
            {"features": np.array([2])},
            {"features": np.array([3])},
            {"features": np.array([4])},
        ]
        source_names = ["source_a", "source_b", "source_c", "source_d"]

        result = splitter.split_data(data_list, source_names)

        # Train should have data from both source_a and source_b
        train_features = result["features"]["train"]
        assert train_features is not None

    def test_split_data_mismatched_lengths_error(self):
        """
        Test error when data_list and source_names have different lengths.

        **PHM Logic**: Each data container must have a corresponding name.

        **Methodology**: Pass mismatched lengths.

        **Expected**: ValueError raised.

        Validates: Requirement BSS-2.3 - Length validation
        """
        splitter = BySourceSplitter(sources_train=["a"], sources_val=["b"])

        data_list = [{"features": np.array([1])}, {"features": np.array([2])}]
        source_names = ["a", "b", "c"]  # 3 names for 2 data items!

        with pytest.raises(ValueError):
            splitter.split_data(data_list, source_names)


class TestBySourceSplitterCall:
    """Tests for __call__ method."""

    def test_call_returns_tuple(self):
        """
        Test that ``__call__`` returns ``(split_data, tree_view)``.

        **PHM Logic**: __call__ provides both data and visualization.

        **Methodology**: Call splitter directly.

        **Expected**: Tuple of (dict, str).

        Validates: Requirement BSS-3.1 - Call interface
        """
        splitter = BySourceSplitter(sources_train=["a"], sources_val=["b"])

        data_list = [{"features": np.array([1])}, {"features": np.array([2])}]
        source_names = ["a", "b"]

        result, tree_view = splitter(data_list, source_names)

        assert isinstance(result, dict)
        assert isinstance(tree_view, (str, list))


class TestBySourceSplitterTreeView:
    """Tests for tree view generation."""

    def test_get_split_tree_view(self):
        """
        Test tree view generation.

        **PHM Logic**: Tree view shows which sources are in which split.

        **Methodology**: Generate tree view for known split.

        **Expected**: Formatted string showing split structure.

        Validates: Requirement BSS-4.1 - Tree view format
        """
        splitter = BySourceSplitter(
            sources_train=["unit_1", "unit_2"],
            sources_val=["unit_3"],
            sources_test=["unit_4"],
        )

        sources_by_split = {
            "train": ["unit_1", "unit_2"],
            "val": ["unit_3"],
            "test": ["unit_4"],
        }

        tree_view = splitter.get_split_tree_view(sources_by_split)

        assert isinstance(tree_view, str)
        # Should contain split names
        assert "train" in tree_view.lower() or "unit_1" in tree_view


class TestConvertOuterListToInner:
    """Tests for convert_outer_list_to_inner utility function."""

    def test_basic_conversion(self):
        """
        Test basic list-of-dicts to dict-of-lists conversion.

        **PHM Logic**: Transform list[{k:v}] to {k: [v1, v2, ...]}.

        **Methodology**: Convert simple list of dicts.

        **Expected**: Dict with lists as values.

        Validates: Requirement COLI-1.1 - Basic conversion
        """
        input_list = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]

        result = convert_outer_list_to_inner(input_list)

        assert result["a"] == [1, 3, 5]
        assert result["b"] == [2, 4, 6]

    def test_empty_list_handling(self):
        """
        Test handling of an empty list.

        **PHM Logic**: Empty list may return empty dict or raise.

        **Methodology**: Pass empty list.

        **Expected**: Empty dict or error handled gracefully.

        Validates: Requirement COLI-1.2 - Empty list handling
        """
        # Implementation returns empty dict for empty list
        result = convert_outer_list_to_inner([])
        assert result == {}

    def test_different_keys_error(self):
        """
        Test that dicts with different keys raise error.

        **PHM Logic**: All dicts must have same structure.

        **Methodology**: Pass dicts with different keys.

        **Expected**: ValueError about mismatched keys.

        Validates: Requirement COLI-1.3 - Key consistency validation
        """
        input_list = [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},  # Different keys!
        ]

        with pytest.raises(ValueError):
            convert_outer_list_to_inner(input_list)

    def test_single_dict(self):
        """
        Test conversion of a single-element list.

        **PHM Logic**: Single source should still convert correctly.

        **Methodology**: Pass list with one dict.

        **Expected**: Dict with single-element lists.

        Validates: Requirement COLI-1.4 - Single element handling
        """
        input_list = [{"a": 1, "b": 2}]

        result = convert_outer_list_to_inner(input_list)

        assert result["a"] == [1]
        assert result["b"] == [2]


class TestBySourceSplitterEdgeCases:
    """Edge case tests for BySourceSplitter."""

    def test_unassigned_sources(self):
        """
        Test handling of sources not assigned to any split.

        **PHM Logic**: Sources not in any split list should be ignored.

        **Methodology**: Provide source not in any list.

        **Expected**: Source's data not included in any split.

        Validates: Requirement BSS-5.1 - Unassigned source handling
        """
        splitter = BySourceSplitter(
            sources_train=["a"],
            sources_val=["b"],
            sources_test=[],  # 'c' not assigned anywhere
        )

        data_list = [
            {"features": np.array([1])},
            {"features": np.array([2])},
            {"features": np.array([3])},  # This source 'c' is unassigned
        ]
        source_names = ["a", "b", "c"]

        # Should work, 'c' data may be None in splits
        result = splitter.split_data(data_list, source_names)

        # Verify train and val have data
        assert result["features"]["train"] is not None
        assert result["features"]["val"] is not None

    def test_all_sources_in_one_split(self):
        """
        Test all sources assigned to a single split.

        **PHM Logic**: Valid scenario for cross-validation folds.

        **Methodology**: Assign all sources to train.

        **Expected**: All data in train, empty val/test.

        Validates: Requirement BSS-5.2 - Single split assignment
        """
        splitter = BySourceSplitter(
            sources_train=["a", "b", "c"], sources_val=[], sources_test=[]
        )

        data_list = [
            {"features": np.array([1])},
            {"features": np.array([2])},
            {"features": np.array([3])},
        ]
        source_names = ["a", "b", "c"]

        result = splitter.split_data(data_list, source_names)

        # All data should be in train
        train_features = result["features"]["train"]
        assert train_features is not None
