from pathlib import Path

import numpy as np
import torch
import pytest

from picid.model.estimators.xgboost.wrapper import FitPredictXGBoostWrapper


def _build_linear_dataset(num_samples: int, num_features: int, *, seed: int = 0):
    torch.manual_seed(seed)
    features = torch.randn(num_samples, num_features)
    targets = features.sum(dim=1)
    return features, targets


def test_xgboost_wrapper_reports_canonical_module():
    assert (
        FitPredictXGBoostWrapper.__module__
        == "picid.model.estimators.xgboost.wrapper"
    )


def test_regression_predict_output_is_two_dimensional():
    X, y = _build_linear_dataset(num_samples=8, num_features=3, seed=0)
    wrapper = FitPredictXGBoostWrapper(
        task_type="regression",
        n_estimators=5,
        random_state=0,
    )

    wrapper.fit(X, y)
    predictions = wrapper.predict(X)

    assert predictions.shape == (X.shape[0], 1)
    assert predictions.dtype == torch.float32


def test_classification_predict_proba_sums_to_one():
    X, raw_y = _build_linear_dataset(num_samples=10, num_features=4, seed=1)
    y = (raw_y > 0).long()
    wrapper = FitPredictXGBoostWrapper(
        task_type="classification",
        n_estimators=5,
        random_state=0,
        num_classes=2,
    )

    wrapper.fit(X, y)
    probabilities = wrapper.predict(X)

    assert probabilities.shape == (X.shape[0], 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(X.shape[0]), atol=1e-5)
    assert probabilities.min() >= 0
    assert probabilities.max() <= 1


def test_serialization_and_loading_uses_model_cache(tmp_path):
    X, y = _build_linear_dataset(num_samples=6, num_features=2, seed=2)
    wrapper = FitPredictXGBoostWrapper(
        task_type="regression",
        n_estimators=5,
        random_state=0,
        model_cache_path=str(tmp_path),
    )

    wrapper.fit(X, y)
    task_id = "xgb-task"
    model_file = wrapper.serialize_model(task_id)

    assert Path(model_file).parent == tmp_path
    assert Path(model_file).exists()
    loaded_model = wrapper.load_model(task_id)

    original_preds = wrapper.backbone.predict(X.numpy())
    loaded_preds = loaded_model.predict(X.numpy())
    assert np.allclose(original_preds, loaded_preds)


def test_invalid_task_type_is_rejected():
    with pytest.raises(ValueError):
        FitPredictXGBoostWrapper(task_type="forecasting")


def test_allows_multi_target_is_false():
    w = FitPredictXGBoostWrapper(task_type="regression", n_estimators=3, random_state=0)
    assert w.allows_multi_target is False


def test_serialize_model_task_id_none_raises():
    w = FitPredictXGBoostWrapper(task_type="regression", n_estimators=3, random_state=0)
    with pytest.raises(ValueError, match="no task_id"):
        w.serialize_model(None)


def test_load_model_task_id_none_raises():
    w = FitPredictXGBoostWrapper(task_type="regression", n_estimators=3, random_state=0)
    with pytest.raises(ValueError, match="no task_id"):
        w.load_model(None)


def test_relative_model_cache_path_is_not_duplicated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative_cache = Path("xgb-rel-cache")
    X, y = _build_linear_dataset(num_samples=8, num_features=2, seed=3)
    w = FitPredictXGBoostWrapper(
        task_type="regression",
        n_estimators=4,
        random_state=0,
        model_cache_path=str(relative_cache),
    )
    w.fit(X, y)
    path = w.serialize_model("xgb-task")
    expected = relative_cache / "xgb-task.joblib"
    assert Path(path) == expected
    assert expected.exists()
    assert not (relative_cache / "xgb-rel-cache" / "xgb-task.joblib").exists()
    loaded = w.load_model("xgb-task")
    assert hasattr(loaded, "predict")
