"""Tests for picid.cli.assemble."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import picid.cli.assemble as assemble
from picid.cli.config_discovery import DatasetGroupInfo, ModelInfo, TaskInfo


class _Prompt:
    def __init__(self, answer):
        self.answer = answer

    def ask(self):
        return self.answer


class _FakeQuestionary:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def select(self, message, choices, default=None):
        self.calls.append(("select", message, choices, default))
        return _Prompt(self.answers.pop(0))

    def text(self, message, default=""):
        self.calls.append(("text", message, default))
        return _Prompt(self.answers.pop(0))

    def Choice(self, title, value=None):
        return SimpleNamespace(title=title, value=title if value is None else value)


def _install_questionary(monkeypatch, answers):
    fake_questionary = _FakeQuestionary(answers)
    monkeypatch.setitem(sys.modules, "questionary", fake_questionary)
    return fake_questionary


def test_main_delegates_to_run_assemble(monkeypatch):
    """main delegates to the interactive assembler."""
    called = []
    monkeypatch.setattr(assemble, "run_assemble", lambda: called.append(True))

    assemble.main()

    assert called == [True]


def test_run_assemble_returns_when_tier_prompt_cancelled(monkeypatch):
    """Cancelling the tier prompt exits early."""
    _install_questionary(monkeypatch, [None])
    task_calls = []
    monkeypatch.setattr(
        assemble, "_funnel_task_first", lambda tier: task_calls.append(tier)
    )

    assemble.run_assemble()

    assert task_calls == []


def test_run_assemble_routes_to_selected_funnel(monkeypatch):
    """The selected top-level mode dispatches to its funnel."""
    _install_questionary(monkeypatch, ["Easy: Use defaults only", "model"])
    model_calls = []
    monkeypatch.setattr(
        assemble, "_funnel_model_first", lambda tier: model_calls.append(tier)
    )
    monkeypatch.setattr(
        assemble,
        "_funnel_task_first",
        lambda tier: (_ for _ in ()).throw(AssertionError("wrong funnel")),
    )
    monkeypatch.setattr(
        assemble,
        "_funnel_dataset_first",
        lambda tier: (_ for _ in ()).throw(AssertionError("wrong funnel")),
    )

    assemble.run_assemble()

    assert model_calls == ["Easy: Use defaults only"]


def test_funnel_task_first_prints_no_data_when_no_tasks(monkeypatch):
    """Task-first flow reports missing task definitions."""
    _install_questionary(monkeypatch, [])
    monkeypatch.setattr(assemble, "list_task_definitions", lambda: [])
    messages = []
    monkeypatch.setattr(assemble, "_no_data", messages.append)

    assemble._funnel_task_first("Easy: Use defaults only")

    assert messages == ["task definitions"]


def test_funnel_task_first_selects_experiment_and_shows_final_command(monkeypatch):
    """Task-first flow lists experiments across groups and forwards the selection."""
    config_root = Path("/tmp/fake-configs")
    task_info = TaskInfo(
        "prognostics",
        "rul",
        config_root / "task_definition" / "prognostics" / "rul.yaml",
    )
    _install_questionary(
        monkeypatch,
        [
            task_info,
            "demo/prognostics/raw/cnn_1d",
        ],
    )
    monkeypatch.setattr(assemble, "CONFIGS_ROOT", config_root)
    monkeypatch.setattr(
        assemble,
        "list_task_definitions",
        lambda: [task_info],
    )
    monkeypatch.setattr(
        assemble,
        "list_experiment_groups",
        lambda: [
            DatasetGroupInfo("demo", Path("demo")),
            DatasetGroupInfo("other", Path("other")),
        ],
    )
    monkeypatch.setattr(
        assemble,
        "list_experiments_for_group_task",
        lambda group, task: [f"{group}/{task}/raw/cnn_1d"],
    )
    monkeypatch.setattr(assemble, "_show_tree_navigation", lambda paths, title: None)
    shown = []
    monkeypatch.setattr(
        assemble,
        "_show_final_command",
        lambda experiment_key, tier: shown.append((experiment_key, tier)),
    )

    assemble._funnel_task_first("Medium: Override paths, debug")

    assert shown == [("demo/prognostics/raw/cnn_1d", "Medium: Override paths, debug")]


def test_funnel_dataset_first_returns_when_no_groups(monkeypatch):
    """Dataset-first flow exits cleanly when no experiment groups exist."""
    _install_questionary(monkeypatch, [])
    monkeypatch.setattr(assemble, "list_experiment_groups", lambda: [])
    messages = []
    monkeypatch.setattr(assemble, "_no_data", messages.append)

    assemble._funnel_dataset_first("Easy: Use defaults only")

    assert messages == ["experiment groups"]


def test_funnel_dataset_first_shows_final_command(monkeypatch, tmp_path):
    """Dataset-first flow drills into group, task, and experiment selection."""
    _install_questionary(
        monkeypatch,
        [
            DatasetGroupInfo("demo", tmp_path / "experiment" / "demo"),
            "forecasting",
            "demo/forecasting/model_a",
        ],
    )
    group_root = tmp_path / "experiment" / "demo" / "forecasting"
    group_root.mkdir(parents=True)
    monkeypatch.setattr(
        assemble,
        "list_experiment_groups",
        lambda: [DatasetGroupInfo("demo", tmp_path / "experiment" / "demo")],
    )
    monkeypatch.setattr(
        assemble,
        "CONFIGS_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        assemble,
        "list_experiments_for_group_task",
        lambda group, task: ["demo/forecasting/model_a"],
    )
    monkeypatch.setattr(assemble, "_show_tree_navigation", lambda paths, title: None)
    shown = []
    monkeypatch.setattr(
        assemble,
        "_show_final_command",
        lambda experiment_key, tier: shown.append((experiment_key, tier)),
    )

    assemble._funnel_dataset_first("Easy: Use defaults only")

    assert shown == [("demo/forecasting/model_a", "Easy: Use defaults only")]


def test_funnel_model_first_filters_matching_experiments(monkeypatch):
    """Model-first flow keeps only experiments that end with the model name."""
    _install_questionary(
        monkeypatch,
        [
            ModelInfo("forecasting", "linear", Path("linear.yaml")),
            "demo/forecasting/raw/linear",
        ],
    )
    monkeypatch.setattr(
        assemble,
        "list_model_configs",
        lambda: [ModelInfo("forecasting", "linear", Path("linear.yaml"))],
    )
    monkeypatch.setattr(
        assemble,
        "list_experiment_groups",
        lambda: [DatasetGroupInfo("demo", Path("demo"))],
    )
    monkeypatch.setattr(
        assemble,
        "list_experiments_for_group_task",
        lambda group, task: [
            "demo/forecasting/raw/linear",
            "demo/forecasting/raw/transformer",
        ],
    )
    monkeypatch.setattr(assemble, "_show_tree_navigation", lambda paths, title: None)
    shown = []
    monkeypatch.setattr(
        assemble,
        "_show_final_command",
        lambda experiment_key, tier: shown.append((experiment_key, tier)),
    )

    assemble._funnel_model_first("Easy: Use defaults only")

    assert shown == [("demo/forecasting/raw/linear", "Easy: Use defaults only")]


def test_parameter_injection_phase_medium_adds_paths_and_debug(monkeypatch):
    """Medium tier can add optional paths and debug overrides."""
    _install_questionary(monkeypatch, ["cluster", "fast"])
    monkeypatch.setattr(assemble, "list_paths_configs", lambda: ["default", "cluster"])
    monkeypatch.setattr(assemble, "list_debug_configs", lambda: ["fast"])

    result = assemble._parameter_injection_phase(
        "demo/prognostics/raw/cnn_1d", "Medium: Override paths, debug"
    )

    assert result == ["paths=cluster", "debug=fast"]


def test_parameter_injection_phase_hard_collects_optimizer_lr_epochs_and_dropout(
    monkeypatch,
):
    """Hard tier adds optimizer, lr, epochs, and cnn-specific dropout overrides."""
    _install_questionary(monkeypatch, ["Yes SGD", "0.02", "50", "0.3"])
    monkeypatch.setattr(
        assemble,
        "get_overridable_from_experiment",
        lambda experiment_key: {"optimization": "default", "model_params": {"lr": 0.1}},
    )
    monkeypatch.setattr(
        assemble, "get_task_type_from_experiment", lambda key: "forecasting"
    )
    monkeypatch.setattr(
        assemble,
        "get_model_specific_params",
        lambda model_name, task_type: {"model": {"dropout_prob": 0.5}},
    )

    result = assemble._parameter_injection_phase(
        "demo/forecasting/raw/cnn_1d",
        "Hard: Override optimizer, learning rate, transforms",
    )

    assert result == [
        "optimization=sgd",
        "optimization.lr=0.02",
        "trainer.max_epochs=50",
        "model.dropout_prob=0.3",
    ]


def test_preview_config_prints_subset_and_clears_hydra(monkeypatch):
    """Config preview composes Hydra config and clears global state afterwards."""
    printed = []
    monkeypatch.setattr(assemble.console, "print", printed.append)
    clear_calls = []
    init_calls = []

    hydra_module = SimpleNamespace(
        initialize=lambda **kwargs: init_calls.append(kwargs),
        compose=lambda **kwargs: {
            "task_definition": {"name": "demo"},
            "model": {"name": "linear"},
            "datasource": {"name": "synthetic"},
            "unused": {"ignored": True},
        },
    )
    global_hydra_module = SimpleNamespace(
        GlobalHydra=type(
            "FakeGlobalHydra",
            (),
            {
                "instance": staticmethod(
                    lambda: SimpleNamespace(clear=lambda: clear_calls.append(True))
                )
            },
        )
    )
    omega_module = SimpleNamespace(
        OmegaConf=SimpleNamespace(
            to_yaml=lambda data: "task_definition:\n  name: demo\n"
        )
    )
    monkeypatch.setitem(sys.modules, "hydra", hydra_module)
    monkeypatch.setitem(sys.modules, "hydra.core.global_hydra", global_hydra_module)
    monkeypatch.setitem(sys.modules, "omegaconf", omega_module)

    assemble._preview_config("demo/prognostics/raw/linear")

    assert init_calls[0]["job_name"] == "assemble_preview"
    assert (
        printed[0]
        == "[bold]Config preview (task_definition, model, datasource):[/bold]"
    )
    assert "task_definition" in printed[1]
    assert clear_calls == [True]


def test_show_final_command_uses_default_paths_and_keeps_paths_first(monkeypatch):
    """Final command always includes paths first, defaulting when absent."""
    printed = []
    monkeypatch.setattr(assemble, "_preview_config", lambda experiment_key: None)
    monkeypatch.setattr(
        assemble,
        "_parameter_injection_phase",
        lambda experiment_key, tier: ["trainer.max_epochs=5", "debug=fast"],
    )
    monkeypatch.setattr(assemble.console, "print", printed.append)

    assemble._show_final_command("demo/prognostics/raw/cnn_1d")

    assert printed[0] == (
        "uv run python picid/run.py "
        "experiment=demo/prognostics/raw/cnn_1d "
        "paths=default trainer.max_epochs=5 debug=fast"
    )
    assert printed[1] == "Copy and run from project root."


def test_no_data_prints_message(capsys):
    """Missing-data helper prints a human-readable message."""
    assemble._no_data("model configs")

    assert capsys.readouterr().out == "No model configs found.\n"


# ---------------------------------------------------------------------------
# _show_tree_navigation
# ---------------------------------------------------------------------------


def test_show_tree_navigation_builds_tree(monkeypatch):
    """_show_tree_navigation prints a tree for a list of paths."""
    printed = []
    monkeypatch.setattr(assemble.console, "print", printed.append)

    assemble._show_tree_navigation(
        ["a/b/c.yaml", "a/b/d.yaml", "x/y.yaml"], title="Root"
    )

    assert len(printed) == 1  # one rich Tree object printed


def test_show_tree_navigation_empty_paths(monkeypatch):
    """_show_tree_navigation with empty list prints empty tree."""
    printed = []
    monkeypatch.setattr(assemble.console, "print", printed.append)

    assemble._show_tree_navigation([], title="Empty")

    assert len(printed) == 1


# ---------------------------------------------------------------------------
# run_assemble — choice=None (second prompt cancelled)
# ---------------------------------------------------------------------------


def test_run_assemble_returns_when_funnel_prompt_cancelled(monkeypatch):
    """After selecting tier, cancelling the funnel prompt exits early (line 89)."""
    _install_questionary(monkeypatch, ["Easy: Use defaults only", None])
    task_calls = []
    monkeypatch.setattr(
        assemble, "_funnel_task_first", lambda tier: task_calls.append(tier)
    )
    monkeypatch.setattr(
        assemble, "_funnel_dataset_first", lambda tier: task_calls.append(tier)
    )
    monkeypatch.setattr(
        assemble, "_funnel_model_first", lambda tier: task_calls.append(tier)
    )

    assemble.run_assemble()

    assert task_calls == []


def test_run_assemble_routes_to_task_funnel(monkeypatch):
    """choice='task' → _funnel_task_first called (line 92)."""
    _install_questionary(monkeypatch, ["Easy: Use defaults only", "task"])
    task_calls = []
    monkeypatch.setattr(
        assemble, "_funnel_task_first", lambda tier: task_calls.append(tier)
    )
    monkeypatch.setattr(assemble, "_funnel_dataset_first", lambda tier: None)
    monkeypatch.setattr(assemble, "_funnel_model_first", lambda tier: None)

    assemble.run_assemble()

    assert task_calls == ["Easy: Use defaults only"]


def test_run_assemble_routes_to_dataset_funnel(monkeypatch):
    """choice='dataset' → _funnel_dataset_first called (line 94)."""
    _install_questionary(monkeypatch, ["Easy: Use defaults only", "dataset"])
    dataset_calls = []
    monkeypatch.setattr(assemble, "_funnel_task_first", lambda tier: None)
    monkeypatch.setattr(
        assemble, "_funnel_dataset_first", lambda tier: dataset_calls.append(tier)
    )
    monkeypatch.setattr(assemble, "_funnel_model_first", lambda tier: None)

    assemble.run_assemble()

    assert dataset_calls == ["Easy: Use defaults only"]


# ---------------------------------------------------------------------------
# _funnel_task_first edge cases
# ---------------------------------------------------------------------------


def test_funnel_task_first_returns_when_task_selection_cancelled(monkeypatch):
    """Cancelling task selection → early return (line 124)."""
    task_info = TaskInfo(
        "prognostics", "rul", Path("/tmp/configs/task_definition/p/r.yaml")
    )
    _install_questionary(monkeypatch, [None])
    monkeypatch.setattr(assemble, "list_task_definitions", lambda: [task_info])
    monkeypatch.setattr(assemble, "CONFIGS_ROOT", Path("/tmp/configs"))
    monkeypatch.setattr(assemble, "_show_tree_navigation", lambda paths, title: None)

    # Should return cleanly without raising
    assemble._funnel_task_first("Easy: Use defaults only")


def test_funnel_task_first_returns_when_no_experiments(monkeypatch):
    """No experiments for task → _no_data + return (lines 132-133)."""
    task_info = TaskInfo(
        "prognostics", "rul", Path("/tmp/configs/task_definition/p/r.yaml")
    )
    _install_questionary(monkeypatch, [task_info])
    monkeypatch.setattr(assemble, "list_task_definitions", lambda: [task_info])
    monkeypatch.setattr(assemble, "CONFIGS_ROOT", Path("/tmp/configs"))
    monkeypatch.setattr(assemble, "_show_tree_navigation", lambda paths, title: None)
    monkeypatch.setattr(assemble, "list_experiment_groups", lambda: [])
    messages = []
    monkeypatch.setattr(assemble, "_no_data", messages.append)

    assemble._funnel_task_first("Easy: Use defaults only")

    assert any("experiment" in m for m in messages)


def test_funnel_task_first_returns_when_experiment_selection_cancelled(monkeypatch):
    """Cancelling experiment selection → early return (line 141)."""
    task_info = TaskInfo(
        "prognostics", "rul", Path("/tmp/configs/task_definition/p/r.yaml")
    )
    group = DatasetGroupInfo("demo", Path("demo"))
    _install_questionary(monkeypatch, [task_info, None])
    monkeypatch.setattr(assemble, "list_task_definitions", lambda: [task_info])
    monkeypatch.setattr(assemble, "CONFIGS_ROOT", Path("/tmp/configs"))
    monkeypatch.setattr(assemble, "_show_tree_navigation", lambda paths, title: None)
    monkeypatch.setattr(assemble, "list_experiment_groups", lambda: [group])
    monkeypatch.setattr(
        assemble,
        "list_experiments_for_group_task",
        lambda g, t: ["demo/prognostics/exp1"],
    )
    shown = []
    monkeypatch.setattr(assemble, "_show_final_command", lambda k, t: shown.append(k))

    assemble._funnel_task_first("Easy: Use defaults only")

    assert shown == []


# ---------------------------------------------------------------------------
# _funnel_dataset_first edge cases
# ---------------------------------------------------------------------------


def test_funnel_dataset_first_returns_when_group_selection_cancelled(monkeypatch):
    """Cancelling group selection → early return (line 167)."""
    group = DatasetGroupInfo("demo", Path("demo"))
    _install_questionary(monkeypatch, [None])
    monkeypatch.setattr(assemble, "list_experiment_groups", lambda: [group])

    assemble._funnel_dataset_first("Easy: Use defaults only")


def test_funnel_dataset_first_returns_when_exp_root_missing(monkeypatch, tmp_path):
    """exp_root doesn't exist → _no_data + return (lines 172-173)."""
    group = DatasetGroupInfo("demo", tmp_path / "experiment" / "demo")
    _install_questionary(monkeypatch, [group])
    monkeypatch.setattr(assemble, "list_experiment_groups", lambda: [group])
    monkeypatch.setattr(assemble, "CONFIGS_ROOT", tmp_path / "nonexistent")
    messages = []
    monkeypatch.setattr(assemble, "_no_data", messages.append)

    assemble._funnel_dataset_first("Easy: Use defaults only")

    assert any("experiment dir" in m for m in messages)


def test_funnel_dataset_first_returns_when_no_task_types(monkeypatch, tmp_path):
    """exp_root exists but empty → _no_data + return (lines 177-178)."""
    exp_root = tmp_path / "experiment" / "demo"
    exp_root.mkdir(parents=True)  # exists but no subdirs
    group = DatasetGroupInfo("demo", exp_root)
    _install_questionary(monkeypatch, [group])
    monkeypatch.setattr(assemble, "list_experiment_groups", lambda: [group])
    monkeypatch.setattr(assemble, "CONFIGS_ROOT", tmp_path)
    messages = []
    monkeypatch.setattr(assemble, "_no_data", messages.append)

    assemble._funnel_dataset_first("Easy: Use defaults only")

    assert any("task type" in m for m in messages)


def test_funnel_dataset_first_returns_when_task_type_cancelled(monkeypatch, tmp_path):
    """Cancelling task type selection → early return (line 182)."""
    exp_root = tmp_path / "experiment" / "demo"
    (exp_root / "forecasting").mkdir(parents=True)
    group = DatasetGroupInfo("demo", exp_root)
    _install_questionary(monkeypatch, [group, None])
    monkeypatch.setattr(assemble, "list_experiment_groups", lambda: [group])
    monkeypatch.setattr(assemble, "CONFIGS_ROOT", tmp_path)

    assemble._funnel_dataset_first("Easy: Use defaults only")


def test_funnel_dataset_first_returns_when_no_experiments(monkeypatch, tmp_path):
    """No experiments for group/task → _no_data + return (lines 186-187)."""
    exp_root = tmp_path / "experiment" / "demo"
    (exp_root / "forecasting").mkdir(parents=True)
    group = DatasetGroupInfo("demo", exp_root)
    _install_questionary(monkeypatch, [group, "forecasting"])
    monkeypatch.setattr(assemble, "list_experiment_groups", lambda: [group])
    monkeypatch.setattr(assemble, "CONFIGS_ROOT", tmp_path)
    monkeypatch.setattr(assemble, "list_experiments_for_group_task", lambda g, t: [])
    messages = []
    monkeypatch.setattr(assemble, "_no_data", messages.append)

    assemble._funnel_dataset_first("Easy: Use defaults only")

    assert any("experiment" in m for m in messages)


def test_funnel_dataset_first_returns_when_experiment_cancelled(monkeypatch, tmp_path):
    """Cancelling experiment selection → early return (line 195)."""
    exp_root = tmp_path / "experiment" / "demo"
    (exp_root / "forecasting").mkdir(parents=True)
    group = DatasetGroupInfo("demo", exp_root)
    _install_questionary(monkeypatch, [group, "forecasting", None])
    monkeypatch.setattr(assemble, "list_experiment_groups", lambda: [group])
    monkeypatch.setattr(assemble, "CONFIGS_ROOT", tmp_path)
    monkeypatch.setattr(
        assemble,
        "list_experiments_for_group_task",
        lambda g, t: ["demo/forecasting/model_a"],
    )
    monkeypatch.setattr(assemble, "_show_tree_navigation", lambda paths, title: None)
    shown = []
    monkeypatch.setattr(assemble, "_show_final_command", lambda k, t: shown.append(k))

    assemble._funnel_dataset_first("Easy: Use defaults only")

    assert shown == []


# ---------------------------------------------------------------------------
# _funnel_model_first edge cases
# ---------------------------------------------------------------------------


def test_funnel_model_first_returns_when_no_models(monkeypatch):
    """No models → _no_data + return (lines 213-214)."""
    _install_questionary(monkeypatch, [])
    monkeypatch.setattr(assemble, "list_model_configs", lambda: [])
    messages = []
    monkeypatch.setattr(assemble, "_no_data", messages.append)

    assemble._funnel_model_first("Easy: Use defaults only")

    assert any("model" in m for m in messages)


def test_funnel_model_first_returns_when_model_selection_cancelled(monkeypatch):
    """Cancelling model selection → early return (line 221)."""
    model = ModelInfo("forecasting", "linear", Path("linear.yaml"))
    _install_questionary(monkeypatch, [None])
    monkeypatch.setattr(assemble, "list_model_configs", lambda: [model])

    assemble._funnel_model_first("Easy: Use defaults only")


def test_funnel_model_first_returns_when_no_matching_experiments(monkeypatch):
    """No experiments match model name → _no_data + return (lines 231-232)."""
    model = ModelInfo("forecasting", "linear", Path("linear.yaml"))
    _install_questionary(monkeypatch, [model])
    monkeypatch.setattr(assemble, "list_model_configs", lambda: [model])
    monkeypatch.setattr(
        assemble,
        "list_experiment_groups",
        lambda: [DatasetGroupInfo("demo", Path("demo"))],
    )
    monkeypatch.setattr(
        assemble,
        "list_experiments_for_group_task",
        lambda g, t: ["demo/forecasting/raw/transformer"],  # no "linear" at end
    )
    messages = []
    monkeypatch.setattr(assemble, "_no_data", messages.append)

    assemble._funnel_model_first("Easy: Use defaults only")

    assert any("model" in m for m in messages)


def test_funnel_model_first_returns_when_experiment_selection_cancelled(monkeypatch):
    """Cancelling experiment selection → early return (line 240)."""
    model = ModelInfo("forecasting", "linear", Path("linear.yaml"))
    _install_questionary(monkeypatch, [model, None])
    monkeypatch.setattr(assemble, "list_model_configs", lambda: [model])
    monkeypatch.setattr(
        assemble,
        "list_experiment_groups",
        lambda: [DatasetGroupInfo("demo", Path("demo"))],
    )
    monkeypatch.setattr(
        assemble,
        "list_experiments_for_group_task",
        lambda g, t: ["demo/forecasting/raw/linear"],
    )
    monkeypatch.setattr(assemble, "_show_tree_navigation", lambda paths, title: None)
    shown = []
    monkeypatch.setattr(assemble, "_show_final_command", lambda k, t: shown.append(k))

    assemble._funnel_model_first("Easy: Use defaults only")

    assert shown == []


# ---------------------------------------------------------------------------
# _parameter_injection_phase — AdamW override (line 300)
# ---------------------------------------------------------------------------


def test_parameter_injection_phase_hard_adamw_override(monkeypatch):
    """Hard tier with 'Yes AdamW' → 'optimization=reduce_on_plateau' added (line 300)."""
    _install_questionary(monkeypatch, ["Yes AdamW", "", ""])
    monkeypatch.setattr(
        assemble,
        "get_overridable_from_experiment",
        lambda key: {},  # no model_params → skips lr prompt
    )
    monkeypatch.setattr(assemble, "get_task_type_from_experiment", lambda key: None)
    monkeypatch.setattr(assemble, "get_model_specific_params", lambda m, t: {})

    result = assemble._parameter_injection_phase(
        "demo/forecasting/raw/cnn_1d",
        "Hard: Override optimizer, learning rate, transforms",
    )

    assert "optimization=reduce_on_plateau" in result
