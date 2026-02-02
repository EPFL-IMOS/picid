from typing import Any, List
from lmetk.evaluator.default import DefaultEvaluator


class ReconstructionEvaluator(DefaultEvaluator):
    """Specialized reconstruction evaluator focused on data management."""

    def __init__(self, metric_names: List[str], **kwargs: Any):
        # We explicitly set task_type to regression for PHM reconstruction tasks
        super().__init__(metric_names=metric_names, task_type="regression", **kwargs)
