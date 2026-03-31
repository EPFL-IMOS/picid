from typing import Any, List
from picid.evaluator.default import DefaultEvaluator


class ReconstructionEvaluator(DefaultEvaluator):
    """
    Specialized reconstruction evaluator focused on data management.

    Parameters
    ----------
    metric_names : List[str]
        Metrics to track.
    **kwargs : Any
        Additional evaluator keyword arguments.
    """

    def __init__(self, metric_names: List[str], **kwargs: Any):
        # We explicitly set task_type to regression for PHM reconstruction tasks
        super().__init__(metric_names=metric_names, task_type="regression", **kwargs)
