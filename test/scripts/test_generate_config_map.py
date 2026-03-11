"""Tests for generate_config_map script.

The script delegates to picid.cli.generate_web. Tests patch config_discovery.CONFIGS_ROOT
to use a tmp_path so we can verify output without the real configs tree.
"""

import json
from pathlib import Path

import pytest

from scripts.experiment_generator.generate_config_map import main


def test_main_writes_json(tmp_path, monkeypatch):
    """main() writes config_map.json with experiment_keys to output_dir."""
    configs = tmp_path / "configs"
    (configs / "experiment" / "phme20" / "prognostics" / "raw").mkdir(parents=True)
    (
        configs / "experiment" / "phme20" / "prognostics" / "raw" / "xgboost.yaml"
    ).write_text("")
    (configs / "paths").mkdir(parents=True)
    (configs / "paths" / "default.yaml").write_text("")
    (configs / "debug").mkdir(parents=True)
    (configs / "debug" / "default.yaml").write_text("")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr("picid.cli.config_discovery.CONFIGS_ROOT", configs)
    main(output_dir=out_dir)

    out_file = out_dir / "config_map.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "experiment_keys" in data
    assert "phme20/prognostics/raw/xgboost" in data["experiment_keys"]
    assert "paths" in data
    assert "default" in data["paths"]
    assert "debug" in data
    assert "default" in data["debug"]
