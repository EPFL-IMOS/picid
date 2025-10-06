from pathlib import Path


def find_config_file(
    experiment_dir: Path, config_name="config.yaml", select_from_hydra=True
) -> Path:
    """Finds the config.yaml file, excluding .hydra directories."""
    # This searches for 'config.yaml' in all subdirectories of experiment_dir
    for config_path in experiment_dir.rglob(config_name):
        # Ensure the config is inside a .hydra metadata folder
        if ".hydra" in config_path.parts and select_from_hydra:
            return config_path

        if not select_from_hydra:
            return config_path

    raise FileNotFoundError(
        f"config.yaml not found in {experiment_dir} (excluding .hydra directories)"
    )
