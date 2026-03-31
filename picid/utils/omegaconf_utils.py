from pathlib import Path


def find_config_file(
    experiment_dir: Path, config_name="config.yaml", select_from_hydra=True
) -> Path:
    """
    Find the resolved config file under an experiment directory.

    Parameters
    ----------
    experiment_dir : Path
        Root directory to search.
    config_name : str, default="config.yaml"
        File name to look for.
    select_from_hydra : bool, default=True
        Whether to prefer paths inside a ``.hydra`` directory.

    Returns
    -------
    Path
        Path to the first matching config file.
    """
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
