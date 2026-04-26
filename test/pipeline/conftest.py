"""
Shared fixtures for pipeline tests.

Provides PHM-realistic mock evaluators, backbones, and batch structures
per docs/dataobject.md, docs/datasets.md (RUL/ConceptRUL/FitPredict batch shapes).
"""

import pytest
import torch
from typing import Dict

from picid.evaluator.base import AbstractEvaluator
from picid.evaluator.default import DefaultEvaluator
from picid.metrics.metric_factory import MetricFactory
from picid.metrics.metrics import MSEMetric
from picid.model.adapters.base import AbstractFitPredictWrapper


# -----------------------------------------------------------------------------
# Gold-standard: nominal PHM batch (RUL / backbone pipeline)
# Docs: dataobject.md - batch contains features/target; datasets.md - RUL batch
# -----------------------------------------------------------------------------


@pytest.fixture
def phm_batch_rul():
    """
    Single batch as produced by RUL/Context datasets for backbone training.

    **PHM Logic**: Features (sensor/time windows), targets (RUL), batch_idx
    for evaluator aggregation. Aligns with docs/datasets.md RULContextBatchDataset.

    **Expected outcome**: Dict usable by BackboneWrapperLightningModule
    training/validation/test steps and process_outputs (batch_idx present).
    """
    B, T, C = 4, 20, 5  # batch, time, channels
    return {
        "features": torch.randn(B, T, C, dtype=torch.float32),
        "rul": torch.rand(B, 1, dtype=torch.float32),
        "batch_idx": torch.arange(B, dtype=torch.int64),
    }


@pytest.fixture
def phm_batch_with_unit_id():
    """
    Batch including unit_id for multi-unit evaluators.

    **PHM Logic**: docs/dataobject.md - multi-unit fleet; unit_id tracks
    which asset each sample belongs to for per-unit metrics.

    **Expected outcome**: process_outputs adds unit_id to model_out when present.
    """
    B = 4
    return {
        "features": torch.randn(B, 10, 3, dtype=torch.float32),
        "rul": torch.rand(B, 1, dtype=torch.float32),
        "batch_idx": torch.arange(B, dtype=torch.int64),
        "unit_id": torch.tensor([1, 1, 2, 2], dtype=torch.int64),
    }


@pytest.fixture
def phm_batch_no_batch_idx():
    """Batch without batch_idx (optional key). process_outputs must not crash."""
    return {
        "features": torch.randn(2, 5, 3),
        "rul": torch.rand(2, 1),
    }


@pytest.fixture
def phm_model_out_tensors():
    """
    Model output as returned by backbone + loss: predictions, targets, loss.
    All tensors for _to_numpy / process_outputs.
    """
    B = 4
    return {
        "predictions": torch.rand(B, 1, 1),
        "targets": torch.rand(B, 1, 1),
        "loss": torch.tensor(0.5),
    }


# -----------------------------------------------------------------------------
# Mock evaluators (AbstractEvaluator interface)
# Docs: evaluators/index.md - update, compute, reset
# -----------------------------------------------------------------------------


class MockEvaluator(AbstractEvaluator):
    """Concrete evaluator that records calls and returns fixed metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._updates = []
        self._reset_count = 0
        self._compute_count = 0

    def update(self, model_out: dict) -> None:
        self._updates.append(model_out)

    def compute(
        self, mode: str, epoch: int = None, step: int = None
    ) -> Dict[str, float]:
        self._compute_count += 1
        return {"mock_metric": 1.0}

    def reset(self) -> None:
        self._reset_count += 1
        self._updates.clear()


@pytest.fixture
def mock_evaluators():
    """Dict of train/val/test evaluators for CustomEvaluatorLightningModule."""
    return {
        "train": MockEvaluator(),
        "val": MockEvaluator(),
        "test": MockEvaluator(),
    }


@pytest.fixture
def phm_default_evaluators(mocker):
    """
    Train/val/test :class:`~picid.evaluator.default.DefaultEvaluator` with real MSE.

    Uses a single MetricFactory patch so each evaluator gets its own metric state.
    """
    mocker.patch.object(
        MetricFactory, "create_metric", lambda name, paths=None: MSEMetric()
    )

    def _make() -> DefaultEvaluator:
        return DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

    return {"train": _make(), "val": _make(), "test": _make()}


# -----------------------------------------------------------------------------
# Mock backbone for BackboneWrapperLightningModule / TrainingLightningModule
# -----------------------------------------------------------------------------


class MockBackbone(torch.nn.Module):
    """Returns dict with predictions, targets; accepts feature shape (B, T, C) with T*C up to 200."""

    def __init__(self, out_features=1, max_features=200):
        super().__init__()
        self.linear = torch.nn.Linear(max_features, out_features)
        self._max_features = max_features

    def forward(self, batch):
        x = batch["features"]
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        dim = x_flat.shape[1]
        if dim < self._max_features:
            pad = torch.zeros(
                B, self._max_features - dim, device=x.device, dtype=x.dtype
            )
            x_flat = torch.cat([x_flat, pad], dim=1)
        else:
            x_flat = x_flat[:, : self._max_features]
        pred = self.linear(x_flat).unsqueeze(1).unsqueeze(2)  # (B,1,1)
        tgt = batch.get("rul", pred.detach())
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(2)
        return {"predictions": pred, "targets": tgt}


class MockLoss(torch.nn.Module):
    """Loss that returns model_out dict with 'loss' key (same contract as picid.loss.default)."""

    def forward(self, model_out=None, batch=None):
        out = dict(model_out) if model_out else {}
        out["loss"] = torch.tensor(0.5, requires_grad=True)
        return out


@pytest.fixture
def mock_backbone():
    return MockBackbone()


@pytest.fixture
def mock_loss():
    return MockLoss()


# -----------------------------------------------------------------------------
# FitPredict batch and backbone (docs/datasets.md FitPredictTaskDataset)
# -----------------------------------------------------------------------------


@pytest.fixture
def phm_fit_predict_batch():
    """
    Batch for FitPredictWrapperLightningModule: context (X), target (y),
    task_idx, optional task_num, task_desc. Batch size 1 for one task per step.
    """
    # (1, n_samples, n_features) and (1, n_samples, n_targets)
    X = torch.randn(1, 50, 5)
    y = torch.randn(1, 50, 1)
    return {
        "context": X,
        "target": y,
        "task_idx": torch.tensor(0),
        "task_num": 3,
        "task_desc": "Task 1 of 3",
    }


class _InnerFitPredictBackbone:
    """Minimal inner model with fit/predict for AbstractFitPredictWrapper(backbone=...)."""

    def __init__(self):
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True

    def predict(self, X):
        assert self._fitted
        return torch.randn(X.shape[0], 1, device=X.device, dtype=X.dtype)


class MockFitPredictBackbone(AbstractFitPredictWrapper):
    """Fit-predict backbone: allows_multi_target=False by default, fit/predict/serialize/load_model."""

    def __init__(self, allows_multi_target=False):
        super().__init__(backbone=_InnerFitPredictBackbone())
        self._allows_multi_target = allows_multi_target
        self._models = {}

    @property
    def allows_multi_target(self):
        return self._allows_multi_target

    def serialize_model(self, model_id):
        self._models[model_id] = True

    def load_model(self, model_id):
        # Pipeline may pass int (single-target) or str (multi-target virtual task)
        key = str(model_id) if not isinstance(model_id, str) else model_id
        assert key in self._models


@pytest.fixture
def mock_fit_predict_backbone():
    return MockFitPredictBackbone(allows_multi_target=False)


@pytest.fixture
def mock_fit_predict_backbone_multi_target():
    return MockFitPredictBackbone(allows_multi_target=True)


@pytest.fixture
def phm_fit_predict_batch_multi_target():
    """Batch with multiple target columns (n_targets>1) for multi-target / virtual-task branch."""
    X = torch.randn(1, 50, 5)
    y = torch.randn(1, 50, 2)  # 2 targets
    return {
        "context": X,
        "target": y,
        "task_idx": torch.tensor(0),
        "task_num": 1,
        "task_desc": "Task 1 of 1",
    }
