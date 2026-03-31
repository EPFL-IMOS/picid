"""Interactive config assembler: select task/dataset/model to build run command."""

from __future__ import annotations

from rich.console import Console
from rich.tree import Tree

from picid.cli.config_discovery import (
    CONFIGS_ROOT,
    DatasetGroupInfo,
    ModelInfo,
    TaskInfo,
    get_task_type_from_experiment,
    list_debug_configs,
    list_experiment_groups,
    list_experiments_for_group_task,
    list_model_configs,
    list_paths_configs,
    list_task_definitions,
)
from picid.cli.parameter_injection import (
    get_model_specific_params,
    get_overridable_from_experiment,
)

console = Console()


def _show_tree_navigation(paths: list[str], title: str = "Configs read") -> None:
    """
    Show which config paths are being read using a Rich tree.

    Parameters
    ----------
    paths : list[str]
        Config paths being traversed.
    title : str, default="Configs read"
        Tree title to display.
    """
    tree = Tree(title)
    # Build tree from path segments: "a/b/c.yaml" -> a -> b -> c.yaml
    roots: dict[str, Tree] = {}
    for path_str in sorted(paths):
        parts = path_str.replace("\\", "/").split("/")
        if not parts:
            continue
        parent = tree
        for i, part in enumerate(parts):
            key = "/".join(parts[: i + 1])
            if key not in roots:
                roots[key] = parent.add(part)
            parent = roots[key]
    console.print(tree)


def main() -> None:
    """Entry point for the ``picid-assemble`` CLI."""
    run_assemble()


def run_assemble() -> None:
    """Run the interactive config assembler with funnel path selection."""
    import questionary

    tier = questionary.select(
        "Configuration complexity:",
        choices=[
            "Easy: Use defaults only",
            "Medium: Override paths, debug",
            "Hard: Override optimizer, learning rate, transforms",
        ],
    ).ask()

    if tier is None:
        return

    choice = questionary.select(
        "How do you want to assemble your experiment?",
        choices=[
            questionary.Choice("Task first (task → experiment)", value="task"),
            questionary.Choice(
                "Dataset first (group → task → experiment)", value="dataset"
            ),
            questionary.Choice("Model first (model → experiment)", value="model"),
        ],
    ).ask()

    if choice is None:
        return

    if choice == "task":
        _funnel_task_first(tier)
    elif choice == "dataset":
        _funnel_dataset_first(tier)
    elif choice == "model":
        _funnel_model_first(tier)


def _funnel_task_first(tier: str) -> None:
    """
    Walk the task-first assembly path.

    Parameters
    ----------
    tier : str
        Requested assembly tier.
    """
    import questionary

    tasks = list_task_definitions()
    if not tasks:
        _no_data("task definitions")
        return

    _show_tree_navigation(
        [str(t.path.relative_to(CONFIGS_ROOT)) for t in tasks],
        title="task_definition/",
    )
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

    _show_tree_navigation(
        [f"experiment/{exp}.yaml" for exp in experiments],
        title="experiment/<group>/<task>/",
    )
    exp_choice = questionary.select("Select experiment", choices=experiments).ask()
    if exp_choice is None:
        return

    _show_final_command(exp_choice, tier)


def _funnel_dataset_first(tier: str) -> None:
    """
    Walk the dataset-first assembly path.

    Parameters
    ----------
    tier : str
        Requested assembly tier.
    """
    import questionary

    groups = list_experiment_groups()
    if not groups:
        _no_data("experiment groups")
        return

    group_choices = [questionary.Choice(g.name, value=g) for g in groups]
    selected_group = questionary.select(
        "Select dataset group", choices=group_choices
    ).ask()
    if selected_group is None:
        return

    group: DatasetGroupInfo = selected_group
    exp_root = CONFIGS_ROOT / "experiment" / group.name
    if not exp_root.exists():
        _no_data(f"experiment dir for group {group.name}")
        return

    task_types = sorted(d.name for d in exp_root.iterdir() if d.is_dir())
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

    _show_tree_navigation(
        [f"experiment/{exp}.yaml" for exp in experiments],
        title=f"experiment/{group.name}/{task_choice}/",
    )
    exp_choice = questionary.select("Select experiment", choices=experiments).ask()
    if exp_choice is None:
        return

    _show_final_command(exp_choice, tier)


def _funnel_model_first(tier: str) -> None:
    """
    Walk the model-first assembly path.

    Parameters
    ----------
    tier : str
        Requested assembly tier.
    """
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

    _show_tree_navigation(
        [f"experiment/{exp}.yaml" for exp in experiments],
        title="experiment/<group>/<task>/",
    )
    exp_choice = questionary.select("Select experiment", choices=experiments).ask()
    if exp_choice is None:
        return

    _show_final_command(exp_choice, tier)


def _parameter_injection_phase(experiment_key: str, tier: str) -> list[str]:
    """
    Prompt for parameter overrides based on the selected tier.

    Parameters
    ----------
    experiment_key : str
        Selected experiment key.
    tier : str
        Requested assembly tier.

    Returns
    -------
    list[str]
        Hydra override strings.
    """
    import questionary

    overrides: list[str] = []

    # Medium tier: paths and debug
    if "Medium" in tier:
        paths_choices = list_paths_configs()
        if paths_choices:
            paths_choice = questionary.select(
                "Paths config:",
                choices=paths_choices,
                default="default",
            ).ask()
            if paths_choice is not None and paths_choice != "default":
                overrides.append(f"paths={paths_choice}")

        debug_choices = ["- None -"] + list_debug_configs()
        if len(debug_choices) > 1:
            debug_choice = questionary.select(
                "Debug (optional):",
                choices=debug_choices,
            ).ask()
            if debug_choice is not None and debug_choice != "- None -":
                overrides.append(f"debug={debug_choice}")

    # Hard tier: optimizer and learning rate
    if "Hard" in tier:
        overridable = get_overridable_from_experiment(experiment_key)

        opt_choice = questionary.select(
            "Override optimizer?",
            choices=[
                "No",
                "Yes AdamW",
                "Yes SGD",
            ],
        ).ask()

        if opt_choice == "Yes AdamW":
            overrides.append("optimization=reduce_on_plateau")
        elif opt_choice == "Yes SGD":
            overrides.append("optimization=sgd")

        if overridable.get("model_params") and "lr" in overridable["model_params"]:
            lr_prompt = questionary.text(
                "Learning rate override (leave empty to keep default):",
                default="",
            ).ask()
            if lr_prompt is not None and lr_prompt.strip():
                overrides.append(f"optimization.lr={lr_prompt.strip()}")

        epochs = questionary.text(
            "trainer.max_epochs (optional, e.g. 100):",
            default="",
        ).ask()
        if epochs is not None and epochs.strip():
            overrides.append(f"trainer.max_epochs={epochs.strip()}")

        # Model-specific params: cnn_1d dropout_prob
        model_name = experiment_key.split("/")[-1]
        task_type = get_task_type_from_experiment(experiment_key)
        if model_name == "cnn_1d" and task_type:
            params = get_model_specific_params("cnn_1d", task_type)
            if "dropout_prob" in params.get("model", {}):
                dropout_val = questionary.text(
                    "dropout_prob (optional):",
                    default="",
                ).ask()
                if dropout_val is not None and dropout_val.strip():
                    overrides.append(f"model.dropout_prob={dropout_val.strip()}")

    return overrides


def _preview_config(experiment_key: str) -> None:
    """
    Compose a config preview without running the experiment.

    Parameters
    ----------
    experiment_key : str
        Selected experiment key.
    """
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    # Hydra requires config_path relative to module __file__; CONFIGS_ROOT = project_root/configs
    hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name="assemble_preview",
    )
    try:
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[f"experiment={experiment_key}"],
        )
        subset = {
            "task_definition": cfg.get("task_definition"),
            "model": cfg.get("model"),
            "datasource": cfg.get("datasource"),
        }
        yaml_str = OmegaConf.to_yaml(subset)
        console.print(
            "[bold]Config preview (task_definition, model, datasource):[/bold]"
        )
        console.print(yaml_str)
    finally:
        GlobalHydra.instance().clear()


def _show_final_command(
    experiment_key: str, tier: str = "Easy: Use defaults only"
) -> None:
    """
    Print the run command for the selected experiment.

    Parameters
    ----------
    experiment_key : str
        Selected experiment key.
    tier : str, default="Easy: Use defaults only"
        Requested assembly tier.
    """
    _preview_config(experiment_key)
    overrides = _parameter_injection_phase(experiment_key, tier)
    cmd_parts = ["uv run python picid/run.py", f"experiment={experiment_key}"]

    paths_override = next((o for o in overrides if o.startswith("paths=")), None)
    cmd_parts.append(paths_override if paths_override else "paths=default")

    for o in overrides:
        if not o.startswith("paths="):
            cmd_parts.append(o)

    cmd = " ".join(cmd_parts)
    console.print(cmd)
    console.print("Copy and run from project root.")


def _no_data(msg: str) -> None:
    """
    Print a message when no data is available.

    Parameters
    ----------
    msg : str
        Short description of the missing data category.
    """
    print(f"No {msg} found.")
