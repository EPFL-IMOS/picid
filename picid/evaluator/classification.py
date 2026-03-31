from typing import Any, List
from picid.evaluator.default import DefaultEvaluator


class ClassificationEvaluator(DefaultEvaluator):
    """
    Specialized classification evaluator.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    metric_names : List[str]
        Metrics to track.
    **kwargs : Any
        Additional evaluator keyword arguments.
    """

    def __init__(self, num_classes: int, metric_names: List[str], **kwargs: Any):
        if num_classes is None or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")

        super().__init__(
            metric_names=metric_names,
            task_type="classification",
            num_classes=num_classes,
            **kwargs,
        )
