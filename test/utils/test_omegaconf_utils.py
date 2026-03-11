"""Tests for picid.utils.omegaconf_utils."""

import pytest

from picid.utils.omegaconf_utils import find_config_file


class TestFindConfigFile:
    def test_finds_config_in_hydra_dir(self, tmp_path):
        hydra_dir = tmp_path / ".hydra"
        hydra_dir.mkdir()
        config_path = hydra_dir / "config.yaml"
        config_path.write_text("test: 1")
        result = find_config_file(
            tmp_path, config_name="config.yaml", select_from_hydra=True
        )
        assert result == config_path

    def test_finds_config_without_hydra_filter(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: 1")
        result = find_config_file(
            tmp_path, config_name="config.yaml", select_from_hydra=False
        )
        assert result == cfg

    def test_raises_when_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config.yaml not found"):
            find_config_file(tmp_path, config_name="config.yaml")
