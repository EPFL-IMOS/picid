from typing import Any, Dict, List, Optional
import numpy as np


class PredictionBuffer:
    """Handles raw data accumulation for saving and plotting."""

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
        """Backward compatibility for existing tests."""
        return self.data["preds"]

    @property
    def targets(self) -> List[np.ndarray]:
        """Backward compatibility for existing tests."""
        return self.data["targets"]

    @property
    def unit_ids(self) -> List[np.ndarray]:
        """Backward compatibility for existing tests."""
        return self.data["unit_ids"]

    def accumulate(
        self, batch: Dict[str, Any], unit_id: Optional[np.ndarray] = None
    ) -> None:
        """Adds a batch of data to internal storage lists."""
        self.data["preds"].append(batch["preds"].copy())
        self.data["targets"].append(batch["targets"].copy())
        if batch.get("is_dual"):
            self.data["norm_preds"].append(batch["norm_preds"].copy())
            self.data["norm_targets"].append(batch["norm_targets"].copy())
        if unit_id is not None:
            self.data["unit_ids"].append(unit_id.copy())

    def get_all(self) -> Dict[str, np.ndarray]:
        """Concatenates all buffered lists into numpy arrays."""
        if not self.data["preds"]:
            return {}
        return {k: np.concatenate(v, axis=0) for k, v in self.data.items() if v}

    def clear(self) -> None:
        """Resets the buffer for a new epoch."""
        for k in self.data:
            self.data[k].clear()
