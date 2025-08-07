
"""LMETK Package with automatic path detection."""

import os
from pathlib import Path

# Your existing imports
# ... other imports ...
import logging
logger = logging.getLogger(__name__)

def _setup_project_root():
    if 'PROJECT_ROOT' in os.environ:
        return
    
    try:
        package_dir = Path(__file__).parent
        project_root = package_dir.parent
        markers = ['pyproject.toml', 'setup.py', 'README.md', 'requirements.txt']
        if any((project_root / marker).exists() for marker in markers):
            os.environ['PROJECT_ROOT'] = str(project_root)
            return
    except Exception as e:
        logger.warning("[WARN] Method 1 failed:", e)
    
    try:
        current = Path.cwd()
        markers = ['pyproject.toml', 'setup.py', 'README.md', 'main.py', 'train.py', '.git']
        for path in [current] + list(current.parents):
            if any((path / marker).exists() for marker in markers):
                os.environ['PROJECT_ROOT'] = str(path)
                logger.info("[INFO] PROJECT_ROOT set (method 2):", os.environ['PROJECT_ROOT'])
                return
    except Exception as e:
        logger.warning("[WARN] Method 2 failed:", e)
    
    os.environ['PROJECT_ROOT'] = str(Path.cwd())
    


# Set up PROJECT_ROOT automatically when package is imported
_setup_project_root()

# Your existing package exports
from . import (
    data,
)

__all__ = [
    "data",
    "_setup_project_root"
    
]


__version__ = "0.0.1"