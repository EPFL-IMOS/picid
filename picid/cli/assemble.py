"""Interactive config assembler: select task/dataset/model to build run command."""

from __future__ import annotations

from picid.cli.config_discovery import (
    CONFIGS_ROOT,
    DatasetGroupInfo,
    ModelInfo,
    TaskInfo,
    list_experiment_groups,
    list_experiments_for_group_task,
    list_model_configs,
    list_task_definitions,
)


def main() -> None:
    """Entry point for picid-assemble CLI."""
    run_assemble()


def run_assemble() -> None:
    """Run the interactive config assembler with funnel path selection."""
    import questionary

    choice = questionary.select(
        "How do you want to assemble your experiment?",
        choices=[
            questionary.Choice("Task first (task → experiment)", value="task"),
            questionary.Choice("Dataset first (group → task → experiment)", value="dataset"),
            questionary.Choice("Model first (model → experiment)", value="model"),
        ],
    ).ask()

    if choice is None:
        return

    if choice == "task":
        _funnel_task_first()
    elif choice == "dataset":
        _funnel_dataset_first()
    elif choice == "model":
        _funnel_model_first()


def _funnel_task_first() -> None:
    """Task → experiments for that task → select experiment → show command."""
    import questionary

    tasks = list_task_definitions()
    if not tasks:
        _no_data("task definitions")
        return

    task_choices = [
        questionary.Choice(f"{t.task_type}/{t.name}", value=t) for t in tasks
    ]
    selected_task = questionary.select("Select task", choices=task_choices).ask()
    if selected_task is None:
        return

    task: TaskInfo = selected_task
    experiments: list[str] = []
    for g in list_experiment_groups():
        experiments.extend(list_experiments_for_group_task(g.name, task.task_type))

    if not experiments:
        _no_data(f"experiments for task {task.task_type}")
        return

    exp_choice = questionary.select("Select experiment", choices=experiments).ask()
    if exp_choice is None:
        return

    _show_final_command(exp_choice)


def _funnel_dataset_first() -> None:
    """Group → task (subdirs under experiment/group/) → experiment → show command."""
    import questionary

    groups = list_experiment_groups()
    if not groups:
        _no_data("experiment groups")
        return

    group_choices = [questionary.Choice(g.name, value=g) for g in groups]
    selected_group = questionary.select("Select dataset group", choices=group_choices).ask()
    if selected_group is None:
        return

    group: DatasetGroupInfo = selected_group
    exp_root = CONFIGS_ROOT / "experiment" / group.name
    if not exp_root.exists():
        _no_data(f"experiment dir for group {group.name}")
        return

    task_types = sorted(
        d.name for d in exp_root.iterdir() if d.is_dir()
    )
    if not task_types:
        _no_data(f"task types under group {group.name}")
        return

    task_choice = questionary.select("Select task type", choices=task_types).ask()
    if task_choice is None:
        return

    experiments = list_experiments_for_group_task(group.name, task_choice)
    if not experiments:
        _no_data(f"experiments for {group.name}/{task_choice}")
        return

    exp_choice = questionary.select("Select experiment", choices=experiments).ask()
    if exp_choice is None:
        return

    _show_final_command(exp_choice)


def _funnel_model_first() -> None:
    """Model → filter experiments using that model → select experiment → show command."""
    import questionary

    models = list_model_configs()
    if not models:
        _no_data("model configs")
        return

    model_choices = [
        questionary.Choice(f"{m.task_type}/{m.model}", value=m) for m in models
    ]
    selected_model = questionary.select("Select model", choices=model_choices).ask()
    if selected_model is None:
        return

    model: ModelInfo = selected_model
    experiments: list[str] = []
    for g in list_experiment_groups():
        for exp in list_experiments_for_group_task(g.name, model.task_type):
            if exp.endswith(f"/{model.model}"):
                experiments.append(exp)

    if not experiments:
        _no_data(f"experiments using model {model.model}")
        return

    exp_choice = questionary.select("Select experiment", choices=experiments).ask()
    if exp_choice is None:
        return

    _show_final_command(exp_choice)


def _show_final_command(experiment_key: str) -> None:
    """Print the run command for the selected experiment."""
    cmd = f"uv run python picid/run.py paths=default experiment={experiment_key}"
    print(cmd)
    print("Copy and run from project root.")


def _no_data(msg: str) -> None:
    """Print a message when no data is available."""
    print(f"No {msg} found.")
