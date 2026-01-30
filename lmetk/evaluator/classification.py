from typing import Dict, List, Optional, Any, override
import numpy as np
from lmetk.evaluator.default import DefaultEvaluator


class ClassificationEvaluator(DefaultEvaluator):
    """
    Lean Classification Evaluator.
    Fixes Keyword Argument Collision and delegates naming policy to base.
    """

    def __init__(self, num_classes: int, metric_names: List[str], **kwargs):
        if num_classes < 1:
            raise ValueError(
                f"num_classes must be a positive integer, got {num_classes}"
            )

        # 1. Clean kwargs to prevent TypeError collision in super().__init__
        kwargs.pop("apply_inverse_scaling", None)
        kwargs.pop("inverse_transform", None)
        kwargs.pop("task_type", None)

        super().__init__(
            metric_names=metric_names,
            num_classes=num_classes,
            task_type="classification",
            apply_inverse_scaling=False,
            inverse_transform=None,
            **kwargs,
        )

    @override
    def _prepare_predictions_for_save(self) -> Dict[str, Any]:
        """Saves raw logits and class labels."""
        if not hasattr(self, "preds") or not self.preds:
            return {}

        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)

        # Uses explicit names for post-hoc diagnostic classification analysis
        return {
            "preds_logits": (["sample", "time", "class_logits"], preds),
            "targets_labels": (["sample", "time", "class_label"], targets),
        }
