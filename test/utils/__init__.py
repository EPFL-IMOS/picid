# Utils tests
# Re-export from the parent utils module for backwards compatibility
import os
from pathlib import Path
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins


class ProjectSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="picid.defaults", path=f"file://{Path.cwd()}/configs"
        )

    def register() -> None:
        """Hydra users should call this function before invoking @hydra.main"""
        print("CWD:", Path.cwd())
        os.environ["PROJECT_ROOT"] = str(Path.cwd())
        Plugins.instance().register(ProjectSearchPathPlugin)
