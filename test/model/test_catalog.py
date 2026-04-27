from pathlib import Path

from picid.model.catalog import (
    LEGACY_MODEL_CONFIG_NAMES,
    MODEL_CATALOG,
    render_model_capabilities_table,
)


def _public_model_config_paths() -> list[Path]:
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "model"
    return sorted(config_dir.glob("*.yaml"))


def _public_model_config_names() -> set[str]:
    return {
        path.stem
        for path in _public_model_config_paths()
        if path.stem not in LEGACY_MODEL_CONFIG_NAMES
    }


def _extract_direct_target(path: Path) -> str | None:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("_target_:"):
            return stripped.split(":", 1)[1].strip()
    return None


def test_model_catalog_covers_public_model_configs():
    catalog_names = {row["config_name"] for row in MODEL_CATALOG}
    assert catalog_names == _public_model_config_names()


def test_model_catalog_entries_are_well_formed():
    required = {
        "config_name",
        "class_path",
        "family",
        "execution_style",
        "tasks",
        "notes",
    }
    valid_task_names = {"regression", "classification", "forecasting"}

    for row in MODEL_CATALOG:
        assert required.issubset(row)
        assert isinstance(row["tasks"], tuple)
        assert row["tasks"]
        assert set(row["tasks"]).issubset(valid_task_names)


def test_model_catalog_entries_are_unique():
    config_names = [row["config_name"] for row in MODEL_CATALOG]
    assert len(config_names) == len(set(config_names))
    assert len(MODEL_CATALOG) == len(
        {
            (
                row["config_name"],
                row["class_path"],
                row["family"],
                row["execution_style"],
                row["tasks"],
            )
            for row in MODEL_CATALOG
        }
    )


def test_model_catalog_direct_targets_match_configs():
    catalog_by_name = {row["config_name"]: row for row in MODEL_CATALOG}
    for path in _public_model_config_paths():
        if path.stem in LEGACY_MODEL_CONFIG_NAMES:
            continue
        direct_target = _extract_direct_target(path)
        if direct_target is None:
            continue
        assert catalog_by_name[path.stem]["class_path"] == direct_target


def test_model_capabilities_table_is_rendered_from_catalog():
    page = Path("docs/modules/modeling/model-capabilities.md").read_text()
    rendered_table = render_model_capabilities_table()
    assert rendered_table in page
