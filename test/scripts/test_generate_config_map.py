"""Tests for generate_config_map script."""
import json
from pathlib import Path

import pytest

from scripts.experiment_generator.generate_config_map import (
    build_config_map,
    crawl_config_group,
    main,
)


def test_crawl_config_group_experiment(tmp_path):
    """Crawl experiment configs returns nested structure."""
    (tmp_path / "phme20" / "prognostics" / "raw").mkdir(parents=True)
    (tmp_path / "phme20" / "prognostics" / "raw" / "xgboost_fit_predict.yaml").write_text("")
    (tmp_path / "phme20" / "prognostics" / "raw" / "cnn_1d.yaml").write_text("")
    result = crawl_config_group(tmp_path, "experiment")
    assert "phme20" in result
    assert "prognostics" in result["phme20"]
    assert "raw" in result["phme20"]["prognostics"]
    opts = result["phme20"]["prognostics"]["raw"]
    assert "xgboost_fit_predict" in opts
    assert "cnn_1d" in opts


def test_crawl_config_group_skips_underscore_files(tmp_path):
    """Files starting with _ are skipped."""
    (tmp_path / "_defaults.yaml").write_text("")
    (tmp_path / "model.yaml").write_text("")
    result = crawl_config_group(tmp_path, "experiment")
    assert "_defaults" not in result
    assert "model" in result


def test_build_config_map_contains_experiment(tmp_path, monkeypatch):
    """build_config_map includes experiment group."""
    configs = tmp_path / "configs"
    (configs / "experiment" / "phme20" / "prognostics" / "raw").mkdir(parents=True)
    (configs / "experiment" / "phme20" / "prognostics" / "raw" / "xgboost.yaml").write_text("")
    (configs / "paths").mkdir(parents=True)
    (configs / "paths" / "default.yaml").write_text("")
    (configs / "debug").mkdir(parents=True)
    (configs / "debug" / "default.yaml").write_text("")
    monkeypatch.chdir(tmp_path)
    out = build_config_map(configs_dir=configs)
    assert "experiment" in out
    assert "phme20" in out["experiment"]
    assert "paths" in out
    assert "default" in out["paths"]
    assert "debug" in out
    assert "default" in out["debug"]


def test_main_writes_json(tmp_path, monkeypatch):
    """main() writes config_map.json with experiment_keys to output_dir."""
    configs = tmp_path / "configs"
    (configs / "experiment" / "phme20" / "prognostics" / "raw").mkdir(parents=True)
    (configs / "experiment" / "phme20" / "prognostics" / "raw" / "xgboost.yaml").write_text("")
    (configs / "paths").mkdir(parents=True)
    (configs / "paths" / "default.yaml").write_text("")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    main(configs_dir=configs, output_dir=out_dir)
    out_file = out_dir / "config_map.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "experiment" in data
    assert "experiment_keys" in data
    assert "phme20/prognostics/raw/xgboost" in data["experiment_keys"]
