# numpydoc ignore=GL08
import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from picid.model.definitions import ALL_TASKS, CLASSIFICATION_TASKS
from picid.evaluator.base import AbstractEvaluator
from picid.evaluator.buffer import PredictionBuffer
from picid.evaluator.hooks.base import BaseEvalHook
from picid.metrics.manager import MetricManager

logger = logging.getLogger(__name__)


class DefaultEvaluator(AbstractEvaluator):
    """
    Orchestrate buffering, metric computation, and hook execution.

    Parameters
    ----------
    metric_names : list[str]
        Metric names tracked by the evaluator.
    task_type : str, default="regression"
        High-level task family used to configure metric handling.
    save_predictions : bool, default=False
        Whether predictions should be buffered for saving.
    collect_predictions : bool | None, default=None
        Explicit override for prediction buffering.
    num_classes : int | None, default=None
        Number of classes for classification tasks.
    paths : Any | None, default=None
        Path container used by downstream logging and export code.
    remote_logger : Any | None, default=None
        External logger used for figures.
    hooks : list[BaseEvalHook] | None, default=None
        Hooks executed after update and compute steps.
    metric_manager : MetricManager | None, default=None
        Optional prebuilt metric manager.
    buffer : PredictionBuffer | None, default=None
        Optional prebuilt prediction buffer.
    **kwargs : Any
        Additional evaluator keyword arguments.
    """

    def __init__(  # numpydoc ignore=GL08
        self,
        metric_names: List[str],
        task_type: str = "regression",
        save_predictions: bool = False,
        collect_predictions: Optional[bool] = None,  # Optional explicit override
        num_classes: Optional[int] = None,
        paths: Optional[Any] = None,
        remote_logger: Optional[Any] = None,
        hooks: Optional[List[BaseEvalHook]] = None,
        metric_manager: Optional[MetricManager] = None,
        buffer: Optional[PredictionBuffer] = None,
        **kwargs: Any,
    ):
        super().__init__(paths=paths, **kwargs)

        # 1. Metric Validation: Ensure the evaluator has a purpose
        if not metric_names or len(metric_names) == 0:
            raise ValueError("metric_names cannot be empty")

        self.task_type = task_type.lower()
        assert self.task_type in ALL_TASKS, f"Unsupported task: {self.task_type}"

        self.save_predictions = save_predictions
        self.remote_logger = remote_logger
        self.num_classes = num_classes
        self.hooks = hooks or []

        # Logic: Automatic Activation vs. Explicit User Override
        if collect_predictions is not None:
            # User explicitly stated True or False, respect the choice.
            self.collect_predictions = collect_predictions
        else:
            # Smart Activation: Enable if we need to save or if hooks are present.
            self.collect_predictions = self.save_predictions or len(self.hooks) > 0

        # Component Initialization
        self.metric_manager = metric_manager or MetricManager(
            metric_names=metric_names,
            task_type=self.task_type,
            paths=self.paths,
            num_classes=self.num_classes,
            is_dual=bool(self.scaling_wrapper.apply_inverse),
        )
        self.buffer = buffer or PredictionBuffer()

    def add_hook(self, hook: BaseEvalHook) -> None:
        """
        Register a new hook and enable prediction collection if needed.

        Parameters
        ----------
        hook : BaseEvalHook
            Hook to append to the evaluator.
        """
        self.hooks.append(hook)
        if not self.collect_predictions:
            logger.info(
                "Automatically enabling collect_predictions due to new hook addition."
            )
            self.collect_predictions = True

    def update(self, model_out: Dict[str, Any]) -> None:
        """
        Update the evaluator with one batch of model outputs.

        Parameters
        ----------
        model_out : dict[str, Any]
            Batch output containing predictions, targets, and optional unit ids.
        """
        if not self._validate_input(model_out):
            return

        batch = self._prepare_batch_data(model_out)

        # Only touch the buffer if the flag is active
        if self.collect_predictions:
            self.buffer.accumulate(batch, model_out.get("unit_id"))

        self.metric_manager.update(
            predictions=batch["preds"],
            targets=batch["targets"],
            norm_preds=batch.get("norm_preds"),
            norm_targets=batch.get("norm_targets"),
        )

        for hook in self.hooks:
            hook.on_update_end(batch, self)

    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """
        Compute metrics and trigger end-of-stage hooks.

        Parameters
        ----------
        mode : str
            Evaluation mode.
        epoch : int
            Current epoch index.
        step : int
            Current step index.

        Returns
        -------
        dict[str, float]
            Computed metric values.
        """
        results = self.metric_manager.compute()
        for hook in self.hooks:
            hook.on_compute_end(results, self, mode, epoch, step)
        return results

    def reset(self) -> None:
        """Reset internal state for a new evaluation run."""
        self.buffer.clear()
        self.metric_manager.reset()

    def log_plot(
        self, fig: plt.Figure, title: str, mode: str, epoch: int, step: int
    ) -> None:
        """
        Log a plot through the configured remote logger.

        Parameters
        ----------
        fig : plt.Figure
            Figure to log.
        title : str
            Human-readable plot title.
        mode : str
            Evaluation mode.
        epoch : int
            Current epoch index.
        step : int
            Current step index.
        """
        if self.remote_logger:
            self.remote_logger.experiment.log({f"plots/{mode}/{title}": fig})

    def _validate_input(  # numpydoc ignore=GL08
        self, model_out: Dict[str, Any]
    ) -> bool:
        p, t = model_out.get("predictions"), model_out.get("targets")
        if p is None or t is None or p.shape[0] == 0:
            return False
        assert t.ndim == 3, "Targets must be 3D (N, T, C)."
        if self.task_type == "forecasting":
            assert (
                p.ndim == 3 and p.shape == t.shape
            ), "Forecasting requires matching (N, T, C) predictions and targets."
            return True
        assert t.shape[-1] == 1, "PHM Constraint: Targets must be (N, T, 1)"
        return True

    def _prepare_batch_data(  # numpydoc ignore=GL08
        self, model_out: Dict[str, Any]
    ) -> Dict[str, Any]:
        preds, targets = model_out["predictions"], model_out["targets"]
        is_dual = self.scaling_wrapper.apply_inverse
        res = {
            "is_dual": is_dual,
            "norm_preds": preds if is_dual else None,
            "norm_targets": targets if is_dual else None,
        }
        if self.task_type in CLASSIFICATION_TASKS:
            p, t = preds, targets.astype(int)
        else:
            p, t = self.scaling_wrapper.inverse_transform_if_needed(
                preds, targets, metadata={"unit_id": model_out.get("unit_id")}
            )
        res.update({"preds": p, "targets": t})
        return res
