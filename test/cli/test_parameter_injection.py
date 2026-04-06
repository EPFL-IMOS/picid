"""Tests for picid.cli.parameter_injection."""

from picid.cli.parameter_injection import (
    _extract_optimization_from_defaults,
    get_model_specific_params,
    get_overridable_from_experiment,
    load_yaml,
)


def test_load_yaml_returns_empty_dict_for_empty_file(tmp_path):
    """Empty YAML files load as an empty dictionary."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert load_yaml(path) == {}


def test_extract_optimization_from_dict_override():
    """Hydra dict-style optimization overrides are detected."""
    defaults = [{"override /optimization": "adamw"}]

    assert _extract_optimization_from_defaults(defaults) == "adamw"


def test_extract_optimization_from_string_override():
    """Hydra string-style optimization overrides are detected."""
    defaults = ["override /optimization: sgd"]

    assert _extract_optimization_from_defaults(defaults) == "sgd"


def test_get_model_specific_params_returns_empty_dict_when_missing(
    tmp_path, monkeypatch
):
    """Missing model config files return an empty dictionary."""
    monkeypatch.setattr(
        "picid.cli.parameter_injection.CONFIGS_ROOT", tmp_path / "configs"
    )

    assert get_model_specific_params("cnn_1d", "prognostics") == {}


def test_get_model_specific_params_reads_existing_model_config(tmp_path, monkeypatch):
    """Existing model configs are loaded from the task-type directory."""
    configs = tmp_path / "configs"
    model_path = configs / "model_configs" / "forecasting"
    model_path.mkdir(parents=True)
    (model_path / "linear.yaml").write_text(
        "model:\n  dropout_prob: 0.25\n", encoding="utf-8"
    )
    monkeypatch.setattr("picid.cli.parameter_injection.CONFIGS_ROOT", configs)

    result = get_model_specific_params("linear", "forecasting")

    assert result == {"model": {"dropout_prob": 0.25}}


def test_get_overridable_from_experiment_returns_defaults_when_missing(
    tmp_path, monkeypatch
):
    """Missing experiment configs fall back to default optimization and params."""
    monkeypatch.setattr(
        "picid.cli.parameter_injection.CONFIGS_ROOT", tmp_path / "configs"
    )

    result = get_overridable_from_experiment("demo/prognostics/raw/cnn_1d")

    assert result == {"optimization": "default", "model_params": {}}


def test_get_overridable_from_experiment_uses_direct_optimization_override(
    tmp_path, monkeypatch
):
    """Direct overrides in the experiment defaults take precedence."""
    configs = tmp_path / "configs"
    experiment_path = configs / "experiment" / "demo" / "prognostics" / "raw"
    optimization_path = configs / "optimization"
    experiment_path.mkdir(parents=True)
    optimization_path.mkdir(parents=True)
    (experiment_path / "cnn_1d.yaml").write_text(
        "defaults:\n  - override /optimization: adamw\n",
        encoding="utf-8",
    )
    (optimization_path / "adamw.yaml").write_text("lr: 0.001\n", encoding="utf-8")
    monkeypatch.setattr("picid.cli.parameter_injection.CONFIGS_ROOT", configs)

    result = get_overridable_from_experiment("demo/prognostics/raw/cnn_1d")

    assert result == {"optimization": "adamw", "model_params": {"lr": 0.001}}


def test_get_overridable_from_experiment_traverses_nested_defaults(
    tmp_path, monkeypatch
):
    """Experiment defaults can resolve optimization through referenced configs."""
    configs = tmp_path / "configs"
    experiment_path = configs / "experiment" / "demo" / "prognostics"
    shared_path = configs / "shared"
    optimization_path = configs / "optimization"
    experiment_path.mkdir(parents=True)
    shared_path.mkdir(parents=True)
    optimization_path.mkdir(parents=True)
    (experiment_path / "base.yaml").write_text(
        "defaults:\n  - /shared/common\n",
        encoding="utf-8",
    )
    (shared_path / "common.yaml").write_text(
        "defaults:\n  - override /optimization: reduce_on_plateau\n",
        encoding="utf-8",
    )
    (optimization_path / "reduce_on_plateau.yaml").write_text(
        "lr: 0.01\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("picid.cli.parameter_injection.CONFIGS_ROOT", configs)

    result = get_overridable_from_experiment("demo/prognostics/base")

    assert result == {
        "optimization": "reduce_on_plateau",
        "model_params": {"lr": 0.01},
    }


def test_get_overridable_from_experiment_omits_lr_when_missing(tmp_path, monkeypatch):
    """Optimization configs without lr do not add model params."""
    configs = tmp_path / "configs"
    experiment_path = configs / "experiment" / "demo"
    optimization_path = configs / "optimization"
    experiment_path.mkdir(parents=True)
    optimization_path.mkdir(parents=True)
    (experiment_path / "base.yaml").write_text(
        "defaults:\n  - override /optimization: sgd\n",
        encoding="utf-8",
    )
    (optimization_path / "sgd.yaml").write_text("momentum: 0.9\n", encoding="utf-8")
    monkeypatch.setattr("picid.cli.parameter_injection.CONFIGS_ROOT", configs)

    result = get_overridable_from_experiment("demo/base")

    assert result == {"optimization": "sgd", "model_params": {}}
