"""PICID package with automatic path detection and lazy top-level imports."""

from importlib import import_module
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _setup_project_root():
    if "PROJECT_ROOT" in os.environ:
        return

    try:
        package_dir = Path(__file__).parent
        project_root = package_dir.parent
        markers = ["pyproject.toml", "setup.py", "README.md", "requirements.txt"]
        if any((project_root / marker).exists() for marker in markers):
            os.environ["PROJECT_ROOT"] = str(project_root)
            return
    except Exception as e:
        logger.warning("[WARN] Method 1 failed:", e)

    try:
        current = Path.cwd()
        markers = [
            "pyproject.toml",
            "setup.py",
            "README.md",
            "main.py",
            "train.py",
            ".git",
        ]
        for path in [current] + list(current.parents):
            if any((path / marker).exists() for marker in markers):
                os.environ["PROJECT_ROOT"] = str(path)
                logger.info(
                    "[INFO] PROJECT_ROOT set (method 2):", os.environ["PROJECT_ROOT"]
                )
                return
    except Exception as e:
        logger.warning("[WARN] Method 2 failed:", e)

    os.environ["PROJECT_ROOT"] = str(Path.cwd())


# Set up PROJECT_ROOT automatically when package is imported
_setup_project_root()


def _silence_wandb_legacy_require_warnings() -> None:
    # Lightning's WandbLogger still calls wandb.require("service") on pickle, which
    # wandb has demoted to a no-op that termwarns on every call. Make the shim silent.
    try:
        from wandb.sdk import wandb_require
    except ImportError:
        return
    wandb_require._Requires.require_service = lambda self: None
    wandb_require._Requires.require_core = lambda self: None


# _silence_wandb_legacy_require_warnings()

__all__ = ["data", "_setup_project_root"]


def __getattr__(name: str):
    if name == "data":
        module = import_module("picid.data")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
