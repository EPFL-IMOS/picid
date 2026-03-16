from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

__all__ = ["Settings", "project_config"]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='TAB_PHM_', extra='ignore')

    local_home: Path = Path(".").expanduser()
    user_home: Path = Path("~").expanduser() / 'TAB_PHM'

    output_dir: Path = user_home

    data_dir: Path = output_dir / "datasets"
    dataset_dir: Path = data_dir

    cache_path: Optional[Path] = output_dir / "cache"
    cache_dir: Optional[Path] = output_dir / "cache"

    log_dir: Path = output_dir / "log_dir"
    plot_dir: Path = output_dir / "plot_dir"
    eval_details: Path= output_dir / "eval_details"
    model_workdir: Path = output_dir / "model_workdir"
    ckpt_dir: Path = output_dir / "eval_details"
    artifacts_dir: Path = output_dir / "artifacts_dir"
    model_cache_dir: Path = output_dir / "model_cache_dir"
    save_dir: Path = output_dir / "save_dir"

    config_path: Path = local_home / "configs"

    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None


project_config = Settings()

def get_config():
    """Read-only access to config"""
    return project_config
