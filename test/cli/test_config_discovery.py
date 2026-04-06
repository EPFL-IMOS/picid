"""Tests for picid.cli.config_discovery."""

from picid.cli.config_discovery import (
    list_debug_configs,
    list_experiment_groups,
    list_experiments_for_group_task,
    list_model_configs,
    list_paths_configs,
    list_task_definitions,
    get_task_type_from_experiment,
)


def test_list_task_definitions_returns_empty_when_root_missing(tmp_path, monkeypatch):
    """Missing task_definition roots return an empty list."""
    monkeypatch.setattr("picid.cli.config_discovery.CONFIGS_ROOT", tmp_path / "configs")

    assert list_task_definitions() == []


def test_list_task_definitions_ignores_non_directory_entries(tmp_path, monkeypatch):
    """Only task-type directories are scanned for task definitions."""
    configs = tmp_path / "configs"
    task_root = configs / "task_definition"
    task_root.mkdir(parents=True)
    (task_root / "README.md").write_text("ignore me", encoding="utf-8")
    prognostics = task_root / "prognostics"
    prognostics.mkdir()
    (prognostics / "rul.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr("picid.cli.config_discovery.CONFIGS_ROOT", configs)

    result = list_task_definitions()

    assert result == [
        ("prognostics", "rul", prognostics / "rul.yaml"),
    ]


def test_list_model_configs_returns_empty_when_root_missing(tmp_path, monkeypatch):
    """Missing model_configs roots return an empty list."""
    monkeypatch.setattr("picid.cli.config_discovery.CONFIGS_ROOT", tmp_path / "configs")

    assert list_model_configs() == []


def test_list_model_configs_filters_task_type_in_isolated_tree(tmp_path, monkeypatch):
    """Task-type filtering works on a synthetic config tree."""
    configs = tmp_path / "configs"
    prognostics = configs / "model_configs" / "prognostics"
    forecasting = configs / "model_configs" / "forecasting"
    prognostics.mkdir(parents=True)
    forecasting.mkdir(parents=True)
    (prognostics / "cnn_1d.yaml").write_text("", encoding="utf-8")
    (forecasting / "linear.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr("picid.cli.config_discovery.CONFIGS_ROOT", configs)

    result = list_model_configs(task_type="forecasting")

    assert result == [
        ("forecasting", "linear", forecasting / "linear.yaml"),
    ]


def test_list_experiment_groups_ignores_files(tmp_path, monkeypatch):
    """Only directories under experiment/ are returned as groups."""
    configs = tmp_path / "configs"
    experiment_root = configs / "experiment"
    experiment_root.mkdir(parents=True)
    (experiment_root / "notes.txt").write_text("ignore me", encoding="utf-8")
    group_dir = experiment_root / "demo"
    group_dir.mkdir()
    monkeypatch.setattr("picid.cli.config_discovery.CONFIGS_ROOT", configs)

    result = list_experiment_groups()

    assert result == [("demo", group_dir)]


def test_list_experiments_for_group_task_builds_nested_keys(tmp_path, monkeypatch):
    """Nested experiment directories are normalized into slash-separated keys."""
    configs = tmp_path / "configs"
    target = configs / "experiment" / "demo" / "forecasting" / "nested"
    target.mkdir(parents=True)
    (target / "model.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr("picid.cli.config_discovery.CONFIGS_ROOT", configs)

    result = list_experiments_for_group_task("demo", "forecasting")

    assert result == ["demo/forecasting/nested/model"]


def test_list_paths_configs_returns_empty_when_root_missing(tmp_path, monkeypatch):
    """Missing paths config roots return an empty list."""
    monkeypatch.setattr("picid.cli.config_discovery.CONFIGS_ROOT", tmp_path / "configs")

    assert list_paths_configs() == []


def test_list_debug_configs_returns_empty_when_root_missing(tmp_path, monkeypatch):
    """Missing debug config roots return an empty list."""
    monkeypatch.setattr("picid.cli.config_discovery.CONFIGS_ROOT", tmp_path / "configs")

    assert list_debug_configs() == []


def test_list_task_definitions_returns_tasks():
    """list_task_definitions returns TaskInfo entries for each task definition."""
    tasks = list_task_definitions()
    assert len(tasks) > 0
    task_types = {t.task_type for t in tasks}
    assert "prognostics" in task_types
    assert "anomaly_detection" in task_types or "forecasting" in task_types
    for t in tasks:
        assert t.task_type
        assert t.name
        assert t.path.exists()
        assert t.path.suffix == ".yaml"


def test_list_model_configs_filtered_by_task_type():
    """list_model_configs returns only models for given task_type when filtered."""
    all_models = list_model_configs(task_type=None)
    prognostics = list_model_configs(task_type="prognostics")
    assert len(prognostics) > 0
    assert all(m.task_type == "prognostics" for m in prognostics)
    assert len(prognostics) <= len(all_models)
    for m in prognostics:
        assert m.model
        assert m.path.exists()
        assert m.path.suffix == ".yaml"


def test_list_experiment_groups_returns_groups():
    """list_experiment_groups returns top-level dirs under experiment/."""
    groups = list_experiment_groups()
    assert len(groups) > 0
    names = {g.name for g in groups}
    assert "unibo" in names
    for g in groups:
        assert g.name
        assert g.path.exists()
        assert g.path.is_dir()


def test_get_task_type_from_experiment():
    """get_task_type_from_experiment extracts task type from full experiment key."""
    assert (
        get_task_type_from_experiment("unibo/prognostics/raw/cnn_1d") == "prognostics"
    )
    assert (
        get_task_type_from_experiment("unibo/prognostics/combined/cnn_1d")
        == "prognostics"
    )
    assert (
        get_task_type_from_experiment(
            "concepts_n_cmapss_ds02/prognostics/ablation/missing_values/cnn_1d"
        )
        == "prognostics"
    )
    assert (
        get_task_type_from_experiment("unibo/anomaly_detection/foo")
        == "anomaly_detection"
    )
    assert get_task_type_from_experiment("group/task") == "task"
    assert get_task_type_from_experiment("only_one") is None
    assert get_task_type_from_experiment("") is None


def test_list_paths_configs_returns_stems():
    """list_paths_configs returns stem of each yaml in configs/paths/."""
    stems = list_paths_configs()
    # May be empty if configs/paths doesn't exist in worktree
    assert isinstance(stems, list)
    assert all(isinstance(s, str) for s in stems)
    if stems:
        assert "default" in stems


def test_list_debug_configs_returns_stems():
    """list_debug_configs returns stem of each yaml in configs/debug/."""
    stems = list_debug_configs()
    assert isinstance(stems, list)
    assert all(isinstance(s, str) for s in stems)


def test_list_experiments_for_group_task_returns_full_keys():
    """list_experiments_for_group_task returns experiment keys including group."""
    experiments = list_experiments_for_group_task("unibo", "prognostics")
    assert len(experiments) > 0
    for key in experiments:
        assert key.startswith("unibo/")
        assert "prognostics" in key
    # Should include variant paths like unibo/prognostics/combined/cnn_1d
    assert any("combined" in k or "raw" in k for k in experiments)
