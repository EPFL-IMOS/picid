"""
Pipeline snapshot tests using phme20 real data only.

Compares current pipeline output to saved slices (loaded and transformed);
asserts preprocessing time within bounds. Uses only phme20—no synthetic data.
Skip when snapshots are missing (@pytest.mark.requires_snapshots).

Fixture generator: test/scripts/snapshot/generate_pipeline_snapshots.py
  Output: test/data/fixtures/pipeline_snapshots/phme20/{loaded,transformed}/

When the pipeline has been modified (transforms, datasource, preprocessing):
  uv run python test/scripts/snapshot/generate_pipeline_snapshots.py

Optional: --output-dir test/data/fixtures/pipeline_snapshots
Optional: --data-dir "$(pwd)/datasets" (when datasets not in default location)

Then commit test/data/fixtures/pipeline_snapshots/.
"""

from __future__ import annotations

from functools import lru_cache
import os
import time
from pathlib import Path

import pytest

from picid.data.preprocessing.preprocessor import PreProcessor
from picid.transforms.base.transform_manager import ConfigTransformManager

# Snapshot fixtures root: test/data/fixtures/pipeline_snapshots
SNAPSHOTS_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / "fixtures" / "pipeline_snapshots"
)

# Keys to extract per config (must match datasource output; phme20 uses "rul" not "target")
CONFIG_KEYS = {
    "phme20": ["features", "rul"],
}


@lru_cache(maxsize=1)
def _ensure_phme20_downloaded(data_dir: str) -> None:
    """
    Ensure the PHME20 dataset is available under ``data_dir/phmd_cache``.

    Parameters
    ----------
    data_dir : str
        Root data directory whose ``phmd_cache`` subdirectory stores PHMD assets.
    """
    from phmd.download import download

    cache_dir = Path(data_dir) / "phmd_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    download("PHME20", cache_dir=str(cache_dir), unzip=True)


def _snapshot_dir(name: str, stage: str) -> Path:
    """
    Return the snapshot directory for the given config and stage.

    Parameters
    ----------
    name : str
        Snapshot-backed datasource name.
    stage : str
        Snapshot stage such as ``loaded`` or ``transformed``.

    Returns
    -------
    pathlib.Path
        Snapshot directory path for the requested dataset stage.
    """
    return SNAPSHOTS_ROOT / name / stage


def _manifest_path(name: str, stage: str) -> Path:
    """
    Return the manifest path for the given config and stage.

    Parameters
    ----------
    name : str
        Snapshot-backed datasource name.
    stage : str
        Snapshot stage such as ``loaded`` or ``transformed``.

    Returns
    -------
    pathlib.Path
        Manifest path for the requested dataset stage.
    """
    return _snapshot_dir(name, stage) / "slices_manifest.json"


def _has_snapshots(name: str) -> bool:
    """
    Return whether both loaded and transformed snapshots are present.

    Parameters
    ----------
    name : str
        Snapshot-backed datasource name.

    Returns
    -------
    bool
        ``True`` when both loaded and transformed manifests exist.
    """
    loaded = _manifest_path(name, "loaded").exists()
    transformed = _manifest_path(name, "transformed").exists()
    return loaded and transformed


@pytest.mark.requires_snapshots
@pytest.mark.parametrize("config_name", ["phme20"])
def test_loaded_slices_match_saved(config_name: str):
    """
    Compare loaded PHME20 slices against saved snapshots.

    Parameters
    ----------
    config_name : str
        Snapshot-backed datasource configuration name.
    """
    if not _has_snapshots(config_name):
        pytest.skip(
            f"Snapshots missing for {config_name}. "
            "Run: uv run python test/scripts/snapshot/generate_pipeline_snapshots.py"
        )
    # Import here so slice_utils is only needed when snapshots exist
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    from test.data.fixtures.pipeline_snapshots.slice_utils import (
        container_to_slice_manifest,
        compare_to_saved_slices,
    )

    CONFIGS = {
        "phme20": (
            "configs/datasource/phme20.yaml",
            "configs/transforms/phme20/normalize_feature_target.yaml",
        ),
    }
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "datasets"
    _ensure_phme20_downloaded(str(data_dir))
    ds_path, _ = CONFIGS[config_name]
    ds_cfg = OmegaConf.load(project_root / ds_path)
    ds_cfg = OmegaConf.merge(
        ds_cfg, OmegaConf.create({"paths": {"data_dir": str(data_dir)}})
    )
    datasource = instantiate(ds_cfg)
    datasource.load_data()
    datasource.split_data()
    container = datasource.get_data()

    loaded_dir = _snapshot_dir(config_name, "loaded")
    saved_path = _manifest_path(config_name, "loaded")
    keys = CONFIG_KEYS.get(config_name, ["features", "target"])
    split_dict = (
        container.to_split_dict() if hasattr(container, "to_split_dict") else {}
    )
    missing = [
        k
        for k in keys
        if not any(k in split_dict.get(s, {}) for s in ("train", "val", "test"))
    ]
    assert not missing, (
        f"Runtime container missing keys {missing} for {config_name}. "
        "Expected PHME20 to be present via datasets/phmd_cache."
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        current_manifest = container_to_slice_manifest(
            container,
            keys=keys,
            out_dir=tmp_path,
            max_rows_per_unit=100,
        )
        compare_to_saved_slices(
            current_manifest,
            saved_path,
            loaded_dir,
            current_split_dict=(
                container.to_split_dict()
                if hasattr(container, "to_split_dict")
                else None
            ),
            current_slices_dir=tmp_path,
        )


@pytest.mark.requires_snapshots
@pytest.mark.parametrize("config_name", ["phme20"])
def test_transformed_slices_match_saved(config_name: str):
    """
    Compare transformed PHME20 slices against saved snapshots.

    Parameters
    ----------
    config_name : str
        Snapshot-backed datasource configuration name.
    """
    if not _has_snapshots(config_name):
        pytest.skip(
            f"Snapshots missing for {config_name}. "
            "Run: uv run python test/scripts/snapshot/generate_pipeline_snapshots.py"
        )
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    from test.data.fixtures.pipeline_snapshots.slice_utils import (
        container_to_slice_manifest,
        compare_to_saved_slices,
        assert_preprocessing_time_within_bounds,
    )

    CONFIGS = {
        "phme20": (
            "configs/datasource/phme20.yaml",
            "configs/transforms/phme20/normalize_feature_target.yaml",
        ),
    }
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "datasets"
    _ensure_phme20_downloaded(str(data_dir))
    ds_path, tf_path = CONFIGS[config_name]
    keys = CONFIG_KEYS.get(config_name, ["features", "target"])
    ds_cfg = OmegaConf.load(project_root / ds_path)
    ds_cfg = OmegaConf.merge(
        ds_cfg, OmegaConf.create({"paths": {"data_dir": str(data_dir)}})
    )
    tf_cfg = OmegaConf.load(project_root / tf_path)
    datasource = instantiate(ds_cfg)
    datasource.load_data()
    datasource.split_data()
    loaded_split_dict = (
        datasource.get_data().to_split_dict()
        if hasattr(datasource.get_data(), "to_split_dict")
        else {}
    )
    missing_loaded = [
        k
        for k in keys
        if not any(k in loaded_split_dict.get(s, {}) for s in ("train", "val", "test"))
    ]
    assert not missing_loaded, (
        f"Loaded runtime container missing keys {missing_loaded} for {config_name}. "
        "Expected PHME20 to be present via datasets/phmd_cache."
    )
    transforms_manager = ConfigTransformManager(transforms_config=tf_cfg)
    preprocessor = PreProcessor(
        datasource=datasource,
        transforms=transforms_manager,
    )
    t0 = time.perf_counter()
    preprocessor.pipeline()
    current_time = time.perf_counter() - t0

    transformed_dir = _snapshot_dir(config_name, "transformed")
    saved_path = _manifest_path(config_name, "transformed")
    keys = CONFIG_KEYS.get(config_name, ["features", "target"])
    split_dict = (
        preprocessor.data.to_split_dict()
        if hasattr(preprocessor.data, "to_split_dict")
        else {}
    )
    missing = [
        k
        for k in keys
        if not any(k in split_dict.get(s, {}) for s in ("train", "val", "test"))
    ]
    assert not missing, (
        f"Transformed runtime container missing keys {missing} for {config_name}. "
        "Expected PHME20 to be present via datasets/phmd_cache."
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        current_manifest = container_to_slice_manifest(
            preprocessor.data,
            keys=keys,
            out_dir=tmp_path,
            max_rows_per_unit=100,
        )
        compare_to_saved_slices(
            current_manifest,
            saved_path,
            transformed_dir,
            current_split_dict=(
                preprocessor.data.to_split_dict()
                if hasattr(preprocessor.data, "to_split_dict")
                else None
            ),
            current_slices_dir=tmp_path,
        )
    max_ratio = float(os.environ.get("PIPELINE_SNAPSHOT_MAX_TIME_RATIO", "5.0"))
    assert_preprocessing_time_within_bounds(
        current_time,
        saved_path,
        max_time_ratio=max_ratio,
    )
