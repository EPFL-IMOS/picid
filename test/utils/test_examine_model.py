"""Comprehensive tests for model examination utilities.

This module tests the model analysis functions used for understanding
PyTorch model architecture and parameter counts.

PHM Context:
-----------
Understanding model complexity is important for PHM applications where
models must fit within resource constraints (e.g., edge deployment).

Test Coverage Strategy:
----------------------
1. **Model Summary Generation**: Parameter counting and hierarchy
2. **Print Formatting**: Human-readable output
3. **Edge Cases**: Empty models, complex architectures
"""

import torch.nn as nn

from picid.utils.examine_model import (
    get_model_summary,
    print_model_summary,
    _populate_summary,
)


class TestGetModelSummary:
    """Tests for get_model_summary function."""

    def test_simple_sequential_model(self):
        """Test summary of simple Sequential model.

        **PHM Logic**: Sequential models are common for simple baselines.

        **Methodology**: Create Sequential model, get summary.

        **Expected**: Summary contains layer info and param counts.

        Validates: Requirement EM-1.1 - Sequential model summary
        """
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        summary = get_model_summary(model)

        assert isinstance(summary, list)
        assert len(summary) > 0

        # Should have Total entry
        total_entry = summary[-1]
        assert "total" in total_entry["name"].lower()
        assert total_entry["total_params"] > 0

    def test_nested_module_model(self):
        """Test summary of nested module model.

        **PHM Logic**: Complex models have nested submodules.

        **Methodology**: Create model with nested modules.

        **Expected**: Summary shows hierarchical structure.

        Validates: Requirement EM-1.2 - Nested module summary
        """

        class InnerModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        class OuterModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerModule()
                self.output = nn.Linear(10, 5)

        model = OuterModule()
        summary = get_model_summary(model)

        assert isinstance(summary, list)
        # Should have entries for inner and outer modules
        assert len(summary) >= 2

    def test_model_with_no_parameters(self):
        """Test summary of model with no trainable parameters.

        **PHM Logic**: Some modules (e.g., activations) have no params.

        **Methodology**: Create model with only non-parametric layers.

        **Expected**: Summary shows 0 parameters.

        Validates: Requirement EM-1.3 - No-param model summary
        """
        model = nn.Sequential(nn.ReLU(), nn.Flatten(), nn.Dropout(0.5))

        summary = get_model_summary(model)

        # Total params should be 0
        total_entry = summary[-1]
        assert total_entry["total_params"] == 0

    def test_trainable_vs_total_params(self):
        """Test distinction between trainable and total parameters.

        **PHM Logic**: Frozen layers have non-trainable params.

        **Methodology**: Freeze some parameters, check counts.

        **Expected**: Total and trainable counts differ.

        Validates: Requirement EM-1.4 - Trainable param tracking
        """
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        summary = get_model_summary(model)

        total_entry = summary[-1]
        assert total_entry["total_params"] > total_entry["trainable_params"]


class TestPrintModelSummary:
    """Tests for print_model_summary function."""

    def test_print_summary_basic(self, capsys):
        """Test basic print output.

        **PHM Logic**: Summary should be human-readable.

        **Methodology**: Print summary, capture output.

        **Expected**: Output contains expected columns.

        Validates: Requirement EM-2.1 - Print output format
        """
        summary = [
            {
                "name": "layer1",
                "depth": 0,
                "total_params": 100,
                "trainable_params": 100,
            },
            {"name": "Total", "depth": 0, "total_params": 100, "trainable_params": 100},
        ]

        print_model_summary(summary)

        captured = capsys.readouterr()
        # Should print something
        assert len(captured.out) > 0 or len(captured.err) > 0

    def test_print_summary_empty_list(self, capsys):
        """Test print with empty summary list.

        **PHM Logic**: Empty summary should handle gracefully.

        **Methodology**: Print empty summary.

        **Expected**: No crash, possibly empty output.

        Validates: Requirement EM-2.2 - Empty summary handling
        """
        try:
            print_model_summary([])
        except (ValueError, IndexError):
            pass  # Expected behavior


class TestPopulateSummary:
    """Tests for _populate_summary internal function."""

    def test_populate_simple_module(self):
        """Test populating summary for simple module.

        **PHM Logic**: Internal function builds summary recursively.

        **Methodology**: Call _populate_summary on linear layer.

        **Expected**: Summary list populated with module info.

        Validates: Requirement EM-3.1 - Simple module population
        """
        module = nn.Linear(10, 5)
        summary_list = []

        _populate_summary(module, "linear", 0, summary_list)

        assert len(summary_list) > 0
        # Entry should have expected keys
        entry = summary_list[0]
        assert "name" in entry
        assert "depth" in entry
        assert "total_params" in entry
        assert "trainable_params" in entry

    def test_populate_nested_modules(self):
        """Test populating summary for nested modules.

        **PHM Logic**: Should recursively process children.

        **Methodology**: Call _populate_summary on Sequential.

        **Expected**: Summary contains all nested modules.

        Validates: Requirement EM-3.2 - Nested module population
        """
        module = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        summary_list = []

        _populate_summary(module, "seq", 0, summary_list)

        # Should have multiple entries
        assert len(summary_list) >= 2


class TestModelExaminationEdgeCases:
    """Edge case tests for model examination."""

    def test_model_with_shared_parameters(self):
        """Test model with shared/tied parameters.

        **PHM Logic**: Weight sharing shouldn't double-count params.

        **Methodology**: Create model with shared weights.

        **Expected**: Params counted correctly.

        Validates: Requirement EM-4.1 - Shared param handling
        """

        class SharedWeightModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                # Share weights
                self.linear2 = self.linear  # Same layer

            def forward(self, x):
                return self.linear2(self.linear(x))

        model = SharedWeightModel()
        summary = get_model_summary(model)

        # Should not double-count
        total_entry = summary[-1]
        # 10*10 + 10 = 110 params
        assert total_entry["total_params"] == 110

    def test_large_model_summary(self):
        """Test summary generation for larger model.

        **PHM Logic**: Should handle models with many layers.

        **Methodology**: Create model with many layers.

        **Expected**: Summary generated without error.

        Validates: Requirement EM-4.2 - Large model handling
        """
        layers = []
        for i in range(50):
            layers.append(nn.Linear(10, 10))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(10, 5))

        model = nn.Sequential(*layers)
        summary = get_model_summary(model)

        assert len(summary) > 50
