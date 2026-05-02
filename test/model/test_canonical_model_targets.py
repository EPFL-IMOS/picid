"""Guard rails for canonical model target paths."""

from __future__ import annotations

from pathlib import Path


def test_canonical_model_configs_target_estimators_namespace():
    for name in [
        "carte_fit_predict",
        "cnn_1d",
        "drift",
        "isolation_forest_fit_predict",
        "linear_regression",
        "mean",
        "mlp",
        "naive",
        "ses",
        "tabdpt_fit_predict",
        "tabpfn_fit_predict",
        "xgboost_fit_predict",
    ]:
        text = Path(f"configs/model/{name}.yaml").read_text(encoding="utf-8")
        assert "picid.model.estimators." in text, name


def test_interface_model_schemas_use_canonical_estimator_paths():
    for rel_path in [
        "picid/interface/schemas/model/cnn1d.py",
        "picid/interface/schemas/model/linear_regression.py",
        "picid/interface/schemas/model/mean.py",
        "picid/interface/schemas/model/mlp.py",
        "picid/interface/schemas/model/naive.py",
    ]:
        text = Path(rel_path).read_text(encoding="utf-8")
        assert "picid.model.estimators." in text, rel_path
