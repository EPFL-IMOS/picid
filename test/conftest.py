# Important to set the environment variable PROJECT_ROOT to the root of the project, hence always import picid first!
# Run tests from repo root with: uv run pytest (see docs/testing/test-entrypoints.md).
import picid  # noqa: F401
import sys
import os
from pathlib import Path

from test.utils import ProjectSearchPathPlugin
import hydra
from hydra.core.global_hydra import GlobalHydra
import pytest

from omegaconf import OmegaConf

# Set PROJECT_ROOT before any Hydra/config usage; infer from test dir if unset
_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", str(_TEST_DIR.parent))
os.environ["PROJECT_ROOT"] = _PROJECT_ROOT
sys.path.insert(0, _PROJECT_ROOT)

N_FEATURES = 5
BATCH_SIZE = 2

OmegaConf.register_new_resolver("infer_dataloader_length", lambda key: 10)
OmegaConf.register_new_resolver("infer_data_dim", lambda key, dim: N_FEATURES)
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))


def pytest_configure(config):
    """Called after command line options have been parsed."""
    ProjectSearchPathPlugin.register()


def pytest_addoption(parser):
    parser.addoption(
        "--paths",
        action="store",
        default="default",
        help="A hydra paths config name to localize testing.",
    )


@pytest.fixture(scope="session", autouse=True)
def hydra_context():
    """Initialize Hydra context once for the entire test session."""
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    with hydra.initialize(
        version_base="1.3", config_path="../configs", job_name="test_run"
    ):
        yield

    # Clean up after tests
    GlobalHydra.instance().clear()


@pytest.fixture(scope="session")
def datasources_names():
    """Return list of configuration names to test."""
    return ["ETTh1", "railway"]


@pytest.fixture(scope="session")
def test_splits():
    """Return list of test splits."""
    return ["train", "val", "test"]


@pytest.fixture(scope="function")
def config_loader(path_cfg):
    """Factory fixture to create hydra configs."""

    def _load_config(config_name: str):
        """Hydra-based config instantiation."""
        cfg = hydra.compose(
            config_name="test.yaml",
            overrides=[
                f"datasource={config_name}",
                f"paths={path_cfg}",
            ],
            return_hydra_config=True,
        )
        return cfg

    return _load_config


@pytest.fixture(scope="function")
def datasource_loader(path_cfg):
    """Factory fixture to create dataset loaders."""

    def _load_dataset(config_name: str):
        """Hydra-based dataset instantiation."""
        cfg = hydra.compose(
            config_name="test.yaml",
            overrides=[
                f"datasource={config_name}",
                f"paths={path_cfg}",
            ],
            return_hydra_config=True,
        )
        data_loader = hydra.utils.instantiate(cfg.datasource)
        data_loader._pipeline()
        return data_loader

    return _load_dataset


@pytest.fixture(scope="session")
def path_cfg(request) -> str:
    ProjectSearchPathPlugin.register()
    """Return the path to the configuration file."""
    return request.config.getoption("--paths")
