"""
Test suite for picid_report.configs.

Validates search space and sort metric configuration functions.
"""

from picid_report.configs import (
    get_search_space,
    get_sort_metric,
    SORT_METRIC_OVERRIDES,
    EXPECTED_SEARCH_SPACE,
)
from picid_report.configs.search_space import get_model_grid_from_search_space
from picid_report.configs.sort_metrics import (
    infer_task_type_from_dataset,
    infer_dataset_category_from_name,
)


# --- get_search_space (configs.search_space.get_search_space) ---


class TestGetSearchSpace:
    """Validates get_search_space: dataset/model lookup."""

    def test_get_search_space_dataset_model_found(self):
        """
        Branch: Dataset and model found in EXPECTED_SEARCH_SPACE.
        Methodology: Add entry to EXPECTED_SEARCH_SPACE, call get_search_space.
        Expected: Returns the search space dict.
        """
        # Temporarily add an entry
        original = EXPECTED_SEARCH_SPACE.copy()
        try:
            EXPECTED_SEARCH_SPACE["test_dataset"] = {
                "test_model": {
                    "task_definition.seq_len": [1, 10],
                    "optimization.lr": [0.001, 0.0005],
                }
            }
            result = get_search_space("test_dataset", "test_model")
            assert result is not None
            assert "task_definition.seq_len" in result
            assert result["task_definition.seq_len"] == [1, 10]
            assert result["optimization.lr"] == [0.001, 0.0005]
        finally:
            EXPECTED_SEARCH_SPACE.clear()
            EXPECTED_SEARCH_SPACE.update(original)

    def test_get_search_space_dataset_not_found(self):
        """
        Branch: Dataset not in EXPECTED_SEARCH_SPACE.
        Methodology: Call with non-existent dataset.
        Expected: Returns None.
        """
        result = get_search_space("nonexistent_dataset", "any_model")
        assert result is None

    def test_get_search_space_model_not_found(self):
        """
        Branch: Dataset found but model not in dataset.
        Methodology: Add dataset entry, call with non-existent model.
        Expected: Returns None.
        """
        original = EXPECTED_SEARCH_SPACE.copy()
        try:
            EXPECTED_SEARCH_SPACE["test_dataset"] = {"test_model": {"lr": [0.001]}}
            result = get_search_space("test_dataset", "nonexistent_model")
            assert result is None
        finally:
            EXPECTED_SEARCH_SPACE.clear()
            EXPECTED_SEARCH_SPACE.update(original)

    def test_get_search_space_empty_config(self):
        """
        Branch: EXPECTED_SEARCH_SPACE is empty.
        Methodology: Ensure empty config, call get_search_space.
        Expected: Returns None.
        """
        original = EXPECTED_SEARCH_SPACE.copy()
        try:
            EXPECTED_SEARCH_SPACE.clear()
            result = get_search_space("any_dataset", "any_model")
            assert result is None
        finally:
            EXPECTED_SEARCH_SPACE.clear()
            EXPECTED_SEARCH_SPACE.update(original)


# --- get_model_grid_from_search_space (configs.search_space.get_model_grid_from_search_space) ---


class TestGetModelGridFromSearchSpace:
    """Validates get_model_grid_from_search_space: single helper for both structures."""

    def test_new_structure_found(self):
        """New structure: dataset -> model -> grid. Returns grid when found."""
        new_space = {
            "ds1": {"model_a": {"lr": [0.01, 0.001], "epochs": [10, 20]}},
        }
        result = get_model_grid_from_search_space("ds1", "model_a", new_space)
        assert result == {"lr": [0.01, 0.001], "epochs": [10, 20]}

    def test_new_structure_dataset_missing(self):
        """New structure: dataset not in space -> None."""
        new_space = {"ds1": {"model_a": {"lr": [0.01]}}}
        result = get_model_grid_from_search_space("other_ds", "model_a", new_space)
        assert result is None

    def test_new_structure_model_missing(self):
        """New structure: model not in dataset -> None."""
        new_space = {"ds1": {"model_a": {"lr": [0.01]}}}
        result = get_model_grid_from_search_space("ds1", "other_model", new_space)
        assert result is None

    def test_legacy_structure_found(self):
        """Legacy structure: model -> grid. Returns grid when found."""
        legacy_space = {"model_a": {"lr": [0.01, 0.001], "epochs": [10, 20]}}
        result = get_model_grid_from_search_space("any_ds", "model_a", legacy_space)
        assert result == {"lr": [0.01, 0.001], "epochs": [10, 20]}

    def test_legacy_structure_model_missing(self):
        """Legacy structure: model not in space -> None."""
        legacy_space = {"model_a": {"lr": [0.01]}}
        result = get_model_grid_from_search_space("any_ds", "other_model", legacy_space)
        assert result is None

    def test_none_search_space(self):
        """search_space None -> None (auto-discovery)."""
        result = get_model_grid_from_search_space("ds1", "model_a", None)
        assert result is None

    def test_empty_search_space(self):
        """search_space {} -> None."""
        result = get_model_grid_from_search_space("ds1", "model_a", {})
        assert result is None


# --- get_sort_metric (configs.sort_metrics.get_sort_metric) ---


class TestGetSortMetric:
    """Validates get_sort_metric: hierarchical metric resolution."""

    def test_get_sort_metric_override_first(self):
        """
        Branch: Specific override takes precedence.
        Methodology: Override exists for (dataset, model).
        Expected: Returns override metric.
        """
        original_overrides = SORT_METRIC_OVERRIDES.copy()
        try:
            SORT_METRIC_OVERRIDES[("test_dataset", "test_model")] = "test/custom"
            result = get_sort_metric(
                "test_dataset",
                "test_model",
                task_type="regression",  # Would normally return "test/mse"
                dataset_category="prognostics",
            )
            assert result == "test/custom"
        finally:
            SORT_METRIC_OVERRIDES.clear()
            SORT_METRIC_OVERRIDES.update(original_overrides)

    def test_get_sort_metric_task_type_default(self):
        """
        Branch: No override, uses task type default.
        Methodology: No override, task_type="regression".
        Expected: Returns task-type default (val_best_rerun/loss from SORT_METRIC_BY_TASK_TYPE).
        """
        original_overrides = SORT_METRIC_OVERRIDES.copy()
        try:
            SORT_METRIC_OVERRIDES.clear()
            result = get_sort_metric(
                "test_dataset",
                "test_model",
                task_type="regression",
                dataset_category="prognostics",
            )
            assert result == "val_best_rerun/loss"
        finally:
            SORT_METRIC_OVERRIDES.clear()
            SORT_METRIC_OVERRIDES.update(original_overrides)

    def test_get_sort_metric_dataset_category_default(self):
        """
        Branch: No override or task type, uses dataset category default.
        Methodology: No override, no task_type, dataset_category="diagnostics".
        Expected: Returns category default (val_best_rerun/loss from SORT_METRIC_BY_DATASET_CATEGORY).
        """
        original_overrides = SORT_METRIC_OVERRIDES.copy()
        try:
            SORT_METRIC_OVERRIDES.clear()
            result = get_sort_metric(
                "test_dataset",
                "test_model",
                task_type=None,
                dataset_category="diagnostics",
            )
            assert result == "val_best_rerun/loss"
        finally:
            SORT_METRIC_OVERRIDES.clear()
            SORT_METRIC_OVERRIDES.update(original_overrides)

    def test_get_sort_metric_fallback_to_global_default(self):
        """
        Branch: No override, no task_type, no dataset_category.
        Methodology: No matches in hierarchy.
        Expected: Returns DEFAULT_SORT_METRIC (val_best_rerun/loss), not None.
        """
        original_overrides = SORT_METRIC_OVERRIDES.copy()
        try:
            SORT_METRIC_OVERRIDES.clear()
            result = get_sort_metric(
                "test_dataset",
                "test_model",
                task_type=None,
                dataset_category=None,
                fallback_to_optimization=True,
            )
            assert result == "val_best_rerun/loss"
        finally:
            SORT_METRIC_OVERRIDES.clear()
            SORT_METRIC_OVERRIDES.update(original_overrides)

    def test_get_sort_metric_classification_task(self):
        """
        Branch: Classification task type.
        Methodology: task_type="classification".
        Expected: Returns task-type default (val_best_rerun/loss).
        """
        original_overrides = SORT_METRIC_OVERRIDES.copy()
        try:
            SORT_METRIC_OVERRIDES.clear()
            result = get_sort_metric(
                "test_dataset",
                "test_model",
                task_type="classification",
            )
            assert result == "val_best_rerun/loss"
        finally:
            SORT_METRIC_OVERRIDES.clear()
            SORT_METRIC_OVERRIDES.update(original_overrides)

    def test_get_sort_metric_fault_classification_task(self):
        """
        Branch: Fault classification task type.
        Methodology: task_type="fault_classification".
        Expected: Returns task-type default (val_best_rerun/loss).
        """
        original_overrides = SORT_METRIC_OVERRIDES.copy()
        try:
            SORT_METRIC_OVERRIDES.clear()
            result = get_sort_metric(
                "test_dataset",
                "test_model",
                task_type="fault_classification",
            )
            assert result == "val_best_rerun/loss"
        finally:
            SORT_METRIC_OVERRIDES.clear()
            SORT_METRIC_OVERRIDES.update(original_overrides)


# --- infer_task_type_from_dataset (configs.sort_metrics.infer_task_type_from_dataset) ---


class TestInferTaskTypeFromDataset:
    """Validates infer_task_type_from_dataset: heuristic task type inference."""

    def test_infer_classification_from_dataset_name(self):
        """
        Branch: Dataset name suggests classification.
        Methodology: Dataset name contains "mzvav" or "diagnostic".
        Expected: Returns "classification".
        """
        assert infer_task_type_from_dataset("mzvav") == "classification"
        assert infer_task_type_from_dataset("hsf15_cooler") == "classification"
        assert infer_task_type_from_dataset("fault_diagnostic") == "classification"

    def test_infer_regression_from_dataset_name(self):
        """
        Branch: Dataset name suggests regression.
        Methodology: Dataset name contains "nb14", "unibo", "cmapss", etc.
        Expected: Returns "regression".
        """
        assert infer_task_type_from_dataset("nb14") == "regression"
        assert infer_task_type_from_dataset("unibo") == "regression"
        assert infer_task_type_from_dataset("concepts_n_cmapss") == "regression"

    def test_infer_unknown_dataset(self):
        """
        Branch: Dataset name doesn't match known patterns.
        Methodology: Unknown dataset name.
        Expected: Returns None.
        """
        result = infer_task_type_from_dataset("unknown_dataset_xyz")
        assert result is None


# --- infer_dataset_category_from_name (configs.sort_metrics.infer_dataset_category_from_name) ---


class TestInferDatasetCategoryFromName:
    """Validates infer_dataset_category_from_name: heuristic category inference."""

    def test_infer_diagnostics_from_dataset_name(self):
        """
        Branch: Dataset name suggests diagnostics.
        Methodology: Dataset name contains "mzvav", "hsf15", "fault", "diagnostic".
        Expected: Returns "diagnostics".
        """
        assert infer_dataset_category_from_name("mzvav") == "diagnostics"
        assert infer_dataset_category_from_name("hsf15_pump") == "diagnostics"
        assert infer_dataset_category_from_name("fault_classification") == "diagnostics"

    def test_infer_prognostics_from_dataset_name(self):
        """
        Branch: Dataset name suggests prognostics.
        Methodology: Dataset name contains "nb14", "unibo", "cmapss", "phme".
        Expected: Returns "prognostics".
        """
        assert infer_dataset_category_from_name("nb14") == "prognostics"
        assert infer_dataset_category_from_name("unibo") == "prognostics"
        assert infer_dataset_category_from_name("concepts_n_cmapss") == "prognostics"
        assert infer_dataset_category_from_name("phme20") == "prognostics"

    def test_infer_unknown_category(self):
        """
        Branch: Dataset name doesn't match known patterns.
        Methodology: Unknown dataset name.
        Expected: Returns None.
        """
        result = infer_dataset_category_from_name("unknown_dataset_xyz")
        assert result is None
