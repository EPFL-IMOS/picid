"""
Config package: dataset/model-level configuration.

- search_space: EXPECTED_SEARCH_SPACE (dataset -> model -> hp -> [values]),
  get_search_space(), get_model_grid_from_search_space() for resolving a grid from
  either the module-level dict or a passed-in dict (new or legacy structure).
- sort_metrics: which metric to use for ranking/selecting best runs (e.g. val_best_rerun/loss);
  get_sort_metric(), inference helpers for task type and dataset category.
"""

from picid_report.configs.search_space import (
    DEFAULT_SEARCH_SPACE,
    EXPECTED_SEARCH_SPACE,
    get_search_space,
    get_model_grid_from_search_space,
)
from picid_report.configs.sort_metrics import (
    SORT_METRIC_BY_TASK_TYPE,
    SORT_METRIC_BY_DATASET_CATEGORY,
    SORT_METRIC_OVERRIDES,
    DEFAULT_SORT_METRIC,
    get_sort_metric,
    infer_task_type_from_dataset,
    infer_dataset_category_from_name,
)

__all__ = [
    "DEFAULT_SEARCH_SPACE",
    "EXPECTED_SEARCH_SPACE",
    "get_search_space",
    "get_model_grid_from_search_space",
    "SORT_METRIC_BY_TASK_TYPE",
    "SORT_METRIC_BY_DATASET_CATEGORY",
    "SORT_METRIC_OVERRIDES",
    "DEFAULT_SORT_METRIC",
    "get_sort_metric",
    "infer_task_type_from_dataset",
    "infer_dataset_category_from_name",
]
