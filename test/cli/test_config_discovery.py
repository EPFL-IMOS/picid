"""Tests for picid.cli.config_discovery."""

from picid.cli.config_discovery import (
    list_experiment_groups,
    list_experiments_for_group_task,
    list_model_configs,
    list_task_definitions,
    get_task_type_from_experiment,
)


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
    assert get_task_type_from_experiment("unibo/prognostics/raw/cnn_1d") == "prognostics"
    assert get_task_type_from_experiment("unibo/prognostics/combined/cnn_1d") == "prognostics"
    assert get_task_type_from_experiment("concepts_n_cmapss_ds02/prognostics/ablation/missing_values/cnn_1d") == "prognostics"
    assert get_task_type_from_experiment("unibo/anomaly_detection/foo") == "anomaly_detection"
    assert get_task_type_from_experiment("group/task") == "task"
    assert get_task_type_from_experiment("only_one") is None
    assert get_task_type_from_experiment("") is None


def test_list_experiments_for_group_task_returns_full_keys():
    """list_experiments_for_group_task returns experiment keys including group."""
    experiments = list_experiments_for_group_task("unibo", "prognostics")
    assert len(experiments) > 0
    for key in experiments:
        assert key.startswith("unibo/")
        assert "prognostics" in key
    # Should include variant paths like unibo/prognostics/combined/cnn_1d
    assert any("combined" in k or "raw" in k for k in experiments)
