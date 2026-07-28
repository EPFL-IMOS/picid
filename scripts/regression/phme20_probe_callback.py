from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


def _tensor_arrays(value: Any, prefix: str = "value") -> dict[str, np.ndarray]:
    """Flatten tensors in a nested batch into stable NumPy-array keys."""
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().contiguous().numpy()}
    if isinstance(value, Mapping):
        arrays: dict[str, np.ndarray] = {}
        for key in sorted(value, key=str):
            arrays.update(_tensor_arrays(value[key], f"{prefix}.{key}"))
        return arrays
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        arrays = {}
        for index, item in enumerate(value):
            arrays.update(_tensor_arrays(item, f"{prefix}.{index}"))
        return arrays
    return {}


def _state_arrays(module: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().contiguous().numpy()
        for name, tensor in module.state_dict().items()
    }


def _array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _array_digest(array),
        }
        for name, array in sorted(arrays.items())
    }


def _loss_value(outputs: Any) -> float | None:
    value = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (float, int)):
        return float(value)
    return None


class PHME20ProbeCallback(Callback):
    """Capture deterministic PHME20 batches and short-training state transitions."""

    def __init__(self, output_dir: str, revision: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.revision = revision
        self.summary: dict[str, Any] = {
            "schema_version": 1,
            "revision": revision,
            "batches": {},
            "initial_parameters": {},
            "train_losses": [],
            "gradient_norms": [],
            "train_steps": {},
            "validation_losses": [],
            "final_parameters": {},
        }

    def _save_arrays(self, filename: str, arrays: Mapping[str, np.ndarray]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_dir / filename, **arrays)

    def _flush(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            dir=self.output_dir, prefix="probe-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.summary, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary_name, self.output_dir / "probe.json")
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def on_fit_start(self, trainer, pl_module) -> None:
        arrays = _state_arrays(pl_module)
        self._save_arrays("initial_parameters.npz", arrays)
        self.summary["initial_parameters"] = _array_manifest(arrays)
        self._flush()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        arrays = _tensor_arrays(batch, "batch")
        filename = f"batch_{batch_idx:03d}.npz"
        self._save_arrays(filename, arrays)
        self.summary["batches"][str(batch_idx)] = _array_manifest(arrays)
        self._flush()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        squared_norm = 0.0
        parameter_count = 0
        for parameter in pl_module.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().double()
            squared_norm += float(torch.sum(gradient * gradient).cpu())
            parameter_count += gradient.numel()
        self.summary["gradient_norms"].append(
            {
                "l2": squared_norm**0.5,
                "parameter_count": parameter_count,
            }
        )
        self._flush()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        loss = _loss_value(outputs)
        self.summary["train_losses"].append(loss)
        arrays = _state_arrays(pl_module)
        filename = f"train_step_{batch_idx:03d}.npz"
        self._save_arrays(filename, arrays)
        self.summary["train_steps"][str(batch_idx)] = _array_manifest(arrays)
        self._flush()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        loss = trainer.callback_metrics.get("val/loss")
        self.summary["validation_losses"].append(_loss_value(loss))
        self._flush()

    def on_fit_end(self, trainer, pl_module) -> None:
        arrays = _state_arrays(pl_module)
        self._save_arrays("final_parameters.npz", arrays)
        self.summary["final_parameters"] = _array_manifest(arrays)
        self._flush()
