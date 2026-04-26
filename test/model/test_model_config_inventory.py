from pathlib import Path

from picid.model.catalog import LEGACY_MODEL_CONFIG_NAMES, MODEL_CATALOG


def _canonical_model_config_names() -> set[str]:
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "model"
    return {
        path.stem
        for path in config_dir.glob("*.yaml")
        if path.stem not in LEGACY_MODEL_CONFIG_NAMES
    }


def test_stale_model_configs_are_not_in_the_canonical_catalog():
    catalog_names = {row["config_name"] for row in MODEL_CATALOG}

    assert LEGACY_MODEL_CONFIG_NAMES.isdisjoint(catalog_names)


def test_canonical_model_configs_match_the_public_inventory():
    catalog_names = {row["config_name"] for row in MODEL_CATALOG}

    assert catalog_names == _canonical_model_config_names()
