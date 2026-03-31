from typing import Any, Dict, List, Optional
import numpy as np


class PredictionBuffer:
    """Store batched predictions, targets, and optional unit identifiers."""

    def __init__(self) -> None:
        self.data: Dict[str, List[np.ndarray]] = {
            "preds": [],
            "targets": [],
            "norm_preds": [],
            "norm_targets": [],
            "unit_ids": [],
        }

    @property
    def preds(self) -> List[np.ndarray]:
        """
        Return buffered predictions.

        Returns
        -------
        list[np.ndarray]
            Buffered prediction arrays.
        """
        return self.data["preds"]

    @property
    def targets(self) -> List[np.ndarray]:
        """
        Return buffered targets.

        Returns
        -------
        list[np.ndarray]
            Buffered target arrays.
        """
        return self.data["targets"]

    @property
    def unit_ids(self) -> List[np.ndarray]:
        """
        Return buffered unit identifiers.

        Returns
        -------
        list[np.ndarray]
            Buffered unit-id arrays.
        """
        return self.data["unit_ids"]

    def accumulate(
        self, batch: Dict[str, Any], unit_id: Optional[np.ndarray] = None
    ) -> None:
        """
        Append one batch of outputs to the internal storage.

        Parameters
        ----------
        batch : dict[str, Any]
            Batch payload containing predictions and targets.
        unit_id : np.ndarray | None, default=None
            Optional unit identifier array aligned with the batch.
        """
        self.data["preds"].append(batch["preds"].copy())
        self.data["targets"].append(batch["targets"].copy())
        if batch.get("is_dual"):
            self.data["norm_preds"].append(batch["norm_preds"].copy())
            self.data["norm_targets"].append(batch["norm_targets"].copy())
        if unit_id is not None:
            self.data["unit_ids"].append(unit_id.copy())

    def get_all(self) -> Dict[str, np.ndarray]:
        """
        Concatenate buffered lists into numpy arrays.

        Returns
        -------
        dict[str, np.ndarray]
            Concatenated buffered arrays keyed by field name.
        """
        if not self.data["preds"]:
            return {}
        return {k: np.concatenate(v, axis=0) for k, v in self.data.items() if v}

    def clear(self) -> None:
        """Resets the buffer for a new epoch."""
        for k in self.data:
            self.data[k].clear()
