import os
from pathlib import Path

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins


class ProjectSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        # Note that foobar/conf is outside of the example plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        search_path.append(
            provider="picid.defaults", path=f"file://{Path.cwd()}/configs"
        )

    def register() -> None:
        """Hydra users should call this function before invoking @hydra.main"""
        os.environ["PROJECT_ROOT"] = str(Path.cwd())
        Plugins.instance().register(ProjectSearchPathPlugin)
