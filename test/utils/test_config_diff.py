"""Tests for picid.utils.config_diff."""

from picid.utils.config_diff import diff_configs


def test_diff_configs_returns_empty_when_identical(tmp_path):
    cfg = tmp_path / "a.yaml"
    cfg.write_text("a: 1\nb: 2\n")
    result = diff_configs(str(cfg), str(cfg))
    assert result["only_in_first"] == []
    assert result["only_in_second"] == []
    assert result["different_values"] == []


def test_diff_configs_reports_different_values(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("trainer:\n  max_epochs: 5\nseed: 42\n")
    b.write_text("trainer:\n  max_epochs: 3\nseed: 42\n")
    result = diff_configs(str(a), str(b))
    assert any(
        k == "trainer.max_epochs" and v1 == 5 and v2 == 3
        for k, v1, v2 in result["different_values"]
    )


def test_diff_configs_reports_only_in_first(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("a: 1\nextra_a: 99\n")
    b.write_text("a: 1\n")
    result = diff_configs(str(a), str(b))
    assert "extra_a" in result["only_in_first"]
    assert result["only_in_second"] == []


def test_diff_configs_reports_only_in_second(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("a: 1\n")
    b.write_text("a: 1\nextra_b: 42\n")
    result = diff_configs(str(a), str(b))
    assert "extra_b" in result["only_in_second"]
    assert result["only_in_first"] == []


def test_diff_configs_nested_flattening(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("model:\n  hidden: 64\n  layers: 2\n")
    b.write_text("model:\n  hidden: 128\n  layers: 2\n")
    result = diff_configs(str(a), str(b))
    diffs = {k: (v1, v2) for k, v1, v2 in result["different_values"]}
    assert "model.hidden" in diffs
    assert diffs["model.hidden"] == (64, 128)
    assert "model.layers" not in diffs
