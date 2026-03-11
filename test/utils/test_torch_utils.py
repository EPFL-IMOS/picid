"""Tests for picid.utils.torch_utils."""

from unittest.mock import patch
import torch
from torch import nn
from omegaconf import OmegaConf

from picid.utils.torch_utils import correct_state_dict_keys, load_model_from_checkpoint


class _ModelWithFC(nn.Module):
    """Minimal model with fc submodule for checkpoint loading tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)


class TestCorrectStateDictKeys:
    def test_removes_backbone_prefix(self):
        state = {
            "backbone.fc.weight": torch.randn(2, 2),
            "backbone.fc.bias": torch.randn(2),
        }
        result = correct_state_dict_keys(state)
        assert "fc.weight" in result
        assert "fc.bias" in result
        assert "backbone.fc.weight" not in result

    def test_leaves_non_backbone_keys(self):
        state = {"fc.weight": torch.randn(2, 2)}
        result = correct_state_dict_keys(state)
        assert "fc.weight" in result


class TestLoadModelFromCheckpoint:
    def test_load_model_from_checkpoint(self, tmp_path):
        ckpt_path = tmp_path / "exp" / "run" / "checkpoint.ckpt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = tmp_path / "config_resolved.yaml"
        OmegaConf.save(
            {"model": {"_target_": "test.utils.test_torch_utils._ModelWithFC"}},
            config_path,
        )
        torch.save(
            {
                "state_dict": {
                    "backbone.fc.weight": torch.randn(2, 2),
                    "backbone.fc.bias": torch.randn(2),
                }
            },
            ckpt_path,
        )
        with patch("picid.utils.torch_utils.find_config_file") as m:
            m.return_value = config_path
            model = load_model_from_checkpoint(ckpt_path, torch.device("cpu"))
        assert isinstance(model, _ModelWithFC)
