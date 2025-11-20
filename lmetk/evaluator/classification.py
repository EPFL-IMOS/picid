from typing import Dict, List, Optional, Any, override
import numpy as np
from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.evaluator.default import DefaultEvaluator


class ClassificationEvaluator(DefaultEvaluator):
    """Evaluator for classification tasks."""

    def __init__(
        self,
        num_classes: int,
        metric_names: List[str],
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        paths: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            metric_names=metric_names,
            num_classes=num_classes,
            task_type="classification",
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            save_predictions=save_predictions,
            paths=paths,
            **kwargs,
        )

    @override
    def _prepare_predictions_for_save(self) -> Dict[str, Any]:
        """
        Custom saving logic for classification: saves logits as predictions
        and ensures targets are properly formatted.
        """
        dict_to_save = {}

        if not self.save_predictions or not self.collect_predictions:
            return dict_to_save

        # Predictions are logits (B, T, L)
        preds = np.concatenate(self.preds, axis=0)
        # Targets (B, T, 1) or (B, T, C)
        targets = np.concatenate(self.targets, axis=0)

        d_names = ["sample"] + ["time"] + ["class_logits"]
        dict_to_save.update({"preds_logits": (d_names, preds)})

        t_names = ["sample"] + ["class_labels"]
        dict_to_save.update({"targets_labels": (t_names, targets)})

        return dict_to_save

    @override
    def _compute_metrics(self) -> Dict[str, float]:
        """
        Computes metrics and strips '_normalized'/'_denormalized' suffixes.
        """
        results = super()._compute_metrics()
        final_results = {}

        for key, value in results.items():
            # 1. Remove aggregation suffix
            base_key = key.removesuffix("_mean")

            # 2. Remove status suffixes
            if base_key.endswith("_normalized"):
                base_key = base_key.removesuffix("_normalized")
            if base_key.endswith("_denormalized"):
                base_key = base_key.removesuffix("_denormalized")

            final_results[base_key] = value

        return final_results
