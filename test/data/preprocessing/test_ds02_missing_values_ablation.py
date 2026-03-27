"""Regression tests for DS02 missing-values ablation transform configs."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from picid.data.data_objects import SplitDatasetContainer
from picid.data.preprocessing.preprocessor import PreProcessor
from picid.transforms.base.transform_manager import ConfigTransformManager


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"
NAN_ALLOWED_TRANSFORMS = (
    "mcar_corruption_train_val",
    "mcar_corruption_test",
    "missing_values_analytics",
    "imputation",
)


class _NoopDatasource:
    """Minimal datasource stub for apply_transforms integration tests."""

    data_name = "noop"
    task_mode = "regression"


def _compose_transforms_config(transform_path: str, output_dir: Path):
    """
    Compose one transform config with deterministic test-only path overrides.

    Parameters
    ----------
    transform_path : str
        Relative Hydra transform config path under ``configs/transforms``.
    output_dir : Path
        Temporary output directory used to resolve ``paths.output_dir``.

    Returns
    -------
    DictConfig
        Composed transforms configuration for the requested path.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="run.yaml",
            overrides=[
                f"transforms={transform_path}",
                "task_definition=prognostics/rul",
                f"paths.output_dir={output_dir}",
                f"paths.work_dir={output_dir}",
            ],
        ).transforms


def _make_ds02_features(seed: int) -> np.ndarray:
    """
    Create a small dense feature matrix compatible with the DS02 ratios list.

    Parameters
    ----------
    seed : int
        Random seed used to build a deterministic feature matrix.

    Returns
    -------
    np.ndarray
        Dense ``(24, 14)`` feature matrix for the DS02 ablation tests.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(24, 14))


def test_ds02_missing_values_ablation_config_disables_validation_for_nan_steps(
    tmp_path,
):
    """
    DS02 missing-values transforms should use the existing validate_output escape hatch.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory used to resolve Hydra output paths.
    """
    transforms_cfg = _compose_transforms_config(
        "concepts_n_cmapss_ds02/ablation_missing_values_depater2023_default",
        tmp_path / "config-default",
    )

    for transform_name in NAN_ALLOWED_TRANSFORMS:
        assert transforms_cfg[transform_name].metadata.validate_output is False


def test_ds02_missing_values_fit_predict_inherits_nan_validation_flags(tmp_path):
    """
    The fit-predict DS02 missing-values stack should inherit the same validation policy.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory used to resolve Hydra output paths.
    """
    transforms_cfg = _compose_transforms_config(
        "concepts_n_cmapss_ds02/ablation_missing_values_depater2023_fit_predict_history",
        tmp_path / "config-fit-predict",
    )

    for transform_name in NAN_ALLOWED_TRANSFORMS:
        assert transforms_cfg[transform_name].metadata.validate_output is False


def test_ds02_missing_values_pipeline_allows_analytics_between_corruption_and_imputation(
    tmp_path,
):
    """
    The DS02 ablation preprocessing chain should allow analytics to observe NaNs.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory used to resolve Hydra output and log paths.
    """
    transforms_cfg = _compose_transforms_config(
        "concepts_n_cmapss_ds02/ablation_missing_values_depater2023_default",
        tmp_path / "pipeline",
    )
    manager = ConfigTransformManager(transforms_config=transforms_cfg)
    transforms = manager.get_data_transforms()
    selected_names = list(NAN_ALLOWED_TRANSFORMS)
    selected_transforms = OrderedDict(
        (name, transforms[name]) for name in selected_names
    )

    data = SplitDatasetContainer(
        features={
            "train": [_make_ds02_features(seed=1)],
            "val": [_make_ds02_features(seed=2)],
            "test": [_make_ds02_features(seed=3)],
        },
    )
    applied_names: list[str] = []
    preprocessor = PreProcessor(datasource=_NoopDatasource())

    processed = preprocessor.apply_transforms(
        data,
        selected_transforms,
        after_each_transform_callback=lambda _data, name: applied_names.append(name),
    )

    assert applied_names == selected_names
    assert "features" in processed
    assert len(processed["features"]["train"]) == 1
    stats_file = (
        tmp_path
        / "pipeline"
        / "logs"
        / "missing_values_stats"
        / "logs"
        / "missing_values_stats.txt"
    )
    assert stats_file.exists()
