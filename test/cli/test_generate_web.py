"""Tests for picid.cli.generate_web."""

from __future__ import annotations

import json

from picid.cli.config_discovery import DatasetGroupInfo
from picid.cli.generate_web import main


def test_main_writes_sorted_unique_config_map_and_excludes_base(tmp_path, monkeypatch):
    """Generated config map excludes base configs and sorts unique experiment keys."""
    group_path = tmp_path / "configs" / "experiment" / "demo"
    (group_path / "forecasting").mkdir(parents=True)
    monkeypatch.setattr(
        "picid.cli.config_discovery.list_experiment_groups",
        lambda: [DatasetGroupInfo(name="demo", path=group_path)],
    )
    monkeypatch.setattr(
        "picid.cli.config_discovery.list_experiments_for_group_task",
        lambda group_name, task_type: [
            "demo/forecasting/base",
            "demo/forecasting/model_b",
            "demo/forecasting/model_a",
            "demo/forecasting/model_b",
        ],
    )
    monkeypatch.setattr("picid.cli.config_discovery.list_paths_configs", lambda: [])
    monkeypatch.setattr(
        "picid.cli.config_discovery.list_debug_configs",
        lambda: ["debug_fast"],
    )

    main(output_dir=tmp_path / "out")

    data = json.loads((tmp_path / "out" / "config_map.json").read_text())
    assert data == {
        "experiment_keys": [
            "demo/forecasting/model_a",
            "demo/forecasting/model_b",
        ],
        "paths": ["default"],
        "debug": ["debug_fast"],
    }


def test_main_creates_output_directory_when_missing(tmp_path, monkeypatch):
    """The output directory is created before writing config_map.json."""
    group_path = tmp_path / "configs" / "experiment" / "demo"
    (group_path / "prognostics").mkdir(parents=True)
    monkeypatch.setattr(
        "picid.cli.config_discovery.list_experiment_groups",
        lambda: [DatasetGroupInfo(name="demo", path=group_path)],
    )
    monkeypatch.setattr(
        "picid.cli.config_discovery.list_experiments_for_group_task",
        lambda group_name, task_type: ["demo/prognostics/raw/cnn_1d"],
    )
    monkeypatch.setattr(
        "picid.cli.config_discovery.list_paths_configs",
        lambda: ["default", "cluster"],
    )
    monkeypatch.setattr("picid.cli.config_discovery.list_debug_configs", lambda: [])

    output_dir = tmp_path / "nested" / "out"
    main(output_dir=output_dir)

    out_file = output_dir / "config_map.json"
    assert out_file.exists()
    assert json.loads(out_file.read_text())["paths"] == ["default", "cluster"]
