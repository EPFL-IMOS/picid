import logging
from typing import Dict, List, Optional, Any, override
import numpy as np
from lmetk.evaluator.default import DefaultEvaluator

logger = logging.getLogger(__name__)


class ClassificationEvaluator(DefaultEvaluator):
    """
    Evaluator specialized for Classification tasks.

    This evaluator extends DefaultEvaluator to handle multiclass metrics and
    formats saved predictions with classification-specific naming conventions
    (logits vs labels) for easier post-hoc analysis.

    Attributes:
        num_classes (int): Number of target classes.
    """

    def __init__(
        self,
        num_classes: int,
        metric_names: List[str],
        save_predictions: bool = False,
        collect_predictions: bool = True,
        paths: Optional[Any] = None,
        remote_logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initializes the ClassificationEvaluator.

        Args:
            num_classes (int): Number of target classes. Must be > 0.
            metric_names (List[str]): List of metrics (e.g. ['accuracy', 'f1']).
            save_predictions (bool): Whether to save predictions to NetCDF.
            collect_predictions (bool): Whether to store batch data in the buffer.
            paths (Optional[Any]): Path configuration object.
            remote_logger (Optional[Any]): Experiment logger (e.g. W&B).
            **kwargs: Additional args passed to DefaultEvaluator.
        """
        if num_classes < 1:
            raise ValueError(
                f"num_classes must be a positive integer, got {num_classes}"
            )

        # LOGIC: Classification tasks operate on logits/labels.
        # Inverse scaling (regression concept) is explicitly disabled.
        kwargs.pop("apply_inverse_scaling", None)
        kwargs.pop("inverse_transform", None)
        kwargs.pop("task_type", None)

        super().__init__(
            metric_names=metric_names,
            task_type="classification",
            num_classes=num_classes,
            apply_inverse_scaling=False,
            inverse_transform=None,
            save_predictions=save_predictions,
            collect_predictions=collect_predictions,
            paths=paths,
            remote_logger=remote_logger,
            **kwargs,
        )

    @override
    def _prepare_predictions_for_save(self) -> Dict[str, Any]:
        """
        Saves raw logits and class labels to a dictionary for NetCDF conversion.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'preds_logits': (dims, array) of raw model outputs.
                - 'targets_labels': (dims, array) of ground truth class indices.
        """
        # Access data from the buffer component
        if not self.buffer.preds:
            return {}

        preds = np.concatenate(self.buffer.preds, axis=0)
        targets = np.concatenate(self.buffer.targets, axis=0)

        # Uses explicit names for post-hoc diagnostic classification analysis
        return {
            "preds_logits": (["sample", "time", "class_logits"], preds),
            "targets_labels": (["sample", "time", "class_label"], targets),
        }
