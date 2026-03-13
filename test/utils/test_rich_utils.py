"""Tests for picid.utils.rich_utils."""

import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from picid.utils.rich_utils import print_config_tree, enforce_tags


class TestPrintConfigTree:
    def test_print_config_tree_builds_tree(self):
        cfg = OmegaConf.create({"data": {"x": 1}, "model": {"y": 2}})
        from lightning_utilities.core.rank_zero import rank_zero_only

        with patch.object(rank_zero_only, "rank", 0):
            with patch("picid.utils.rich_utils.rich.print") as m:
                print_config_tree(cfg)
        m.assert_called_once()

    def test_print_config_tree_save_to_file(self, tmp_path):
        cfg = OmegaConf.create(
            {"data": {"x": 1}, "paths": {"output_dir": str(tmp_path)}}
        )
        from lightning_utilities.core.rank_zero import rank_zero_only

        with patch.object(rank_zero_only, "rank", 0):
            with patch("picid.utils.rich_utils.rich.print"):
                print_config_tree(cfg, save_to_file=True)
        assert (tmp_path / "config_tree.log").exists()


class TestEnforceTags:
    def test_raises_when_multirun_and_no_tags(self):
        cfg = OmegaConf.create({})
        from lightning_utilities.core.rank_zero import rank_zero_only

        mock_hydra = MagicMock()
        mock_hydra.cfg.hydra.job = {"id": "0"}
        with patch.object(rank_zero_only, "rank", 0):
            with patch("picid.utils.rich_utils.HydraConfig", return_value=mock_hydra):
                with pytest.raises(ValueError, match="Specify tags"):
                    enforce_tags(cfg)

    def test_prompts_when_no_tags(self):
        cfg = OmegaConf.create({})
        from lightning_utilities.core.rank_zero import rank_zero_only

        mock_hydra = MagicMock()
        mock_hydra.cfg.hydra.job = {}
        with patch.object(rank_zero_only, "rank", 0):
            with patch("picid.utils.rich_utils.HydraConfig", return_value=mock_hydra):
                with patch("picid.utils.rich_utils.Prompt.ask", return_value="dev"):
                    enforce_tags(cfg)
        assert cfg.tags == ["dev"]
