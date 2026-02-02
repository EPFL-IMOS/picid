from typing import Any, List
from lmetk.evaluator.default import DefaultEvaluator


class ReconstructionEvaluator(DefaultEvaluator):
    """Specialized reconstruction evaluator."""

    def __init__(
        self, metric_names: List[str], plot_reconstructions: bool = False, **kwargs: Any
    ):
        super().__init__(metric_names=metric_names, task_type="regression", **kwargs)
        self.plot_reconstructions = plot_reconstructions
