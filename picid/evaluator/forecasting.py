# numpydoc ignore=GL08
from typing import Any, Dict, List, Optional
from picid.evaluator.default import DefaultEvaluator


class ForecastingEvaluator(DefaultEvaluator):
    """
    Forecasting evaluator with horizon slicing logic.

    Parameters
    ----------
    target_dim_position : Optional[int]
        Target feature index for univariate forecasting (ignored when ``task_mode`` is ``"multivariate"``).
    metric_names : List[str]
        Metrics to track.
    effective_pred_len : Optional[int], default=None
        Tail length used for metric computation.
    model_pred_len : Optional[int], default=None
        Full prediction horizon produced by the model.
    **kwargs : Any
        Additional evaluator keyword arguments.
    """

    def __init__(  # numpydoc ignore=GL08
        self,
        target_dim_position: Optional[int],
        metric_names: List[str],
        effective_pred_len: Optional[int] = None,
        model_pred_len: Optional[int] = None,
        **kwargs: Any,
    ):
        if model_pred_len is None:
            raise ValueError("model_pred_len must be set for ForecastingEvaluator.")

        if effective_pred_len and effective_pred_len > model_pred_len:
            raise ValueError("effective_pred_len cannot be greater than model_pred_len")

        self.model_pred_len = model_pred_len
        self.effective_pred_len = effective_pred_len
        self.target_dim_position = target_dim_position
        self.task_mode = kwargs.get("task_mode", "univariate")

        # Store for test compatibility
        self.model_seq_len = kwargs.get("model_seq_len")
        self.model_label_len = kwargs.get("model_label_len")

        super().__init__(metric_names=metric_names, task_type="forecasting", **kwargs)

    def update(self, model_out: Dict[str, Any]) -> None:  # numpydoc ignore=GL08
        p, t = model_out.get("predictions"), model_out.get("targets")

        # Test compatibility: Shape mismatch check
        if p is not None and t is not None:
            assert p.shape == t.shape, "Shape mismatch between predictions and targets"

            if self.effective_pred_len is not None:
                p, t = (
                    p[:, -self.effective_pred_len :, :],
                    t[:, -self.effective_pred_len :, :],
                )

            if self.task_mode == "univariate" and self.target_dim_position is not None:
                f = self.target_dim_position
                p, t = p[:, :, f : f + 1], t[:, :, f : f + 1]

            model_out["predictions"], model_out["targets"] = p, t

        super().update(model_out)
