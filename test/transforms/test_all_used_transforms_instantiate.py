"""Test that all transform configs used by experiments can instantiate."""

from __future__ import annotations

import tempfile

import hydra
import pytest
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path

from picid.transforms.base.transform_manager import ConfigTransformManager

REPO = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO / "configs"

# Override paths that depend on ${hydra:runtime.output_dir} and ${hydra:runtime.cwd}
# so transforms can instantiate without a full Hydra run context
TEST_OUTPUT_DIR = Path(tempfile.gettempdir()) / "picid_transform_test"
PATHS_OVERRIDES = [
    f"paths.output_dir={TEST_OUTPUT_DIR}",
    f"paths.work_dir={TEST_OUTPUT_DIR}",
]


@pytest.fixture(scope="session", autouse=True)
def hydra_initialized():
    """Ensure Hydra is initialized before each test.

    Other tests (e.g. baselines) may clear GlobalHydra. This fixture guarantees
    a valid context for compose() so tests pass regardless of execution order.
    """
    GlobalHydra.instance().clear()
    # config_path is relative to cwd; from test/transforms/ need ../../configs
    with hydra.initialize(
        version_base="1.3", config_path="../../configs", job_name="test_run"
    ):
        yield


# Transform paths from override /transforms: in configs/experiment/ (unique set)
USED_TRANSFORM_PATHS = [
    "airbus_helicopter/statistics_fit_predict",
    "battery/nb14/combined",
    "battery/nb14/combined_fit_predict",
    "battery/nb14/raw",
    "battery/nb14/raw_fit_predict",
    "battery/unibo/ablation_missing_values_combined",
    "battery/unibo/ablation_missing_values_combined_fit_predict",
    "battery/unibo/anomaly_detection_fit_predict",
    "battery/unibo/combined",
    "battery/unibo/combined_fit_predict",
    "battery/unibo/raw",
    "bearings/pronostia/combined",
    "bearings/pronostia/combined_fit_predict",
    "bearings/pronostia/raw",
    "bearings/pronostia/spectral",
    "bearings/pronostia/spectral_fit_predict",
    "bearings/pronostia/statistics",
    "bearings/pronostia/statistics_fit_predict",
    "bearings/pronostia/stftt",
    "bearings/pronostia/stftt_fit_predict",
    "bearings/xjtu_sy/combined",
    "bearings/xjtu_sy/combined_fit_predict",
    "bearings/xjtu_sy/raw",
    "concepts_n_cmapss/depater2023_default",
    "concepts_n_cmapss/depater2023_fit_predict_history",
    "concepts_n_cmapss_ds02/ablation_missing_values_depater2023_default",
    "concepts_n_cmapss_ds02/ablation_missing_values_depater2023_fit_predict_history",
    "concepts_n_cmapss_ds02/depater2023_default",
    "concepts_n_cmapss_ds02/depater2023_fit_predict_history",
    "concepts_n_cmapss_multi/depater2023_default",
    "concepts_n_cmapss_multi/depater2023_fit_predict_history",
    "hsf15/default",
    "hsf15/statistics_fit_predict",
    "mzvav/default",
    "mzvav/fit_predict_history",
    "phme20/ablation_missing_values",
    "phme20/ablation_missing_values_fit_predict",
    "phme20/ablation_missing_values_fit_predict_nan",
    "phme20/normalize_feature_target",
    "phme20/normalize_feature_target_fit_predict",
    "railway/context_paper",
    "railway/railway_no_leak",
    "railway/railway_seasonal_leak",
]


@pytest.mark.parametrize("transform_path", USED_TRANSFORM_PATHS)
def test_used_transform_config_instantiates(transform_path: str) -> None:
    """Each used transform config loads and instantiates via ConfigTransformManager."""
    config_path = CONFIGS_DIR / "transforms" / f"{transform_path}.yaml"
    if not config_path.exists():
        pytest.skip(f"Transform config not found: {config_path}")

    # Uses Hydra context from test conftest (initialized with config_path="../configs")
    # Provide task_definition for configs that use ${task_definition.seq_len}
    # Override paths so transforms using ${paths.log_dir} etc. resolve without HydraConfig
    overrides = [
        f"transforms={transform_path}",
        "task_definition=prognostics/rul",
        *PATHS_OVERRIDES,
    ]
    cfg = compose(
        config_name="run",
        overrides=overrides,
    )
    transforms_config = cfg.get("transforms")
    if transforms_config is None or (
        hasattr(transforms_config, "keys") and len(list(transforms_config.keys())) == 0
    ):
        pytest.skip(f"Transforms config empty for {transform_path}")

    manager = ConfigTransformManager(transforms_config=transforms_config)
    transforms = manager.get_data_transforms()
    assert len(transforms) > 0, f"No transforms instantiated for {transform_path}"
