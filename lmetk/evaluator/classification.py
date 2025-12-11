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

        t_names = ["sample"] + ["time"] + ["class_label"]
        dict_to_save.update({"targets_labels": (t_names, targets)})

        return dict_to_save

    @override
    def _compute_metrics(self) -> Dict[str, float]:
        """
        Computes metrics and strips '_normalized'/'_denormalized' suffixes.
        """
        results = super()._compute_metrics()
        final_results = {}

        print(results, "results in classification evaluator")
        for key, value in results.items():
            # 1. Remove aggregation suffix (does nothing if suffix is not present)
            base_key = key.removesuffix("_mean")

            # 2. Remove status suffixes
            if key.endswith("_normalized"):
                base_key = key.removesuffix("_normalized")
            elif key.endswith("_denormalized"):
                base_key = key.removesuffix("_denormalized")
            else:
                raise ValueError(f"Unexpected metric key format: {key}")

            # Final key is just the base name (e.g., 'accuracy')
            # If a base metric appears twice (e.g., normalized and denormalized),
            # we must decide which one to keep, or if both should be kept under
            # different names (if scaling was actually applied).

            # Since denormalized results are nonsensical in CLASSIFICATION,
            # we prioritize the BASE result which implicitly uses the final labels.

            # We will use the base name as the final key

            if base_key in final_results:
                assert (
                    final_results[base_key] == value
                ), f"Conflict in metric values for {base_key}: {final_results[base_key]} vs {value}"
            else:
                final_results[base_key] = value

        return final_results
