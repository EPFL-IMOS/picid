"""Slow real-data regressions for the split-view contract used by run.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra

from picid.data.data_objects import SplitViewPolicy
from picid.data.preprocessing.preprocessor import PreProcessor
from picid.transforms.base.transform_manager import ConfigTransformManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@pytest.mark.slow
@pytest.mark.real_data
@pytest.mark.parametrize(
    "experiment_name",
    [
        "phme20/prognostics/raw/lstm",
        "nb14/prognostics/combined/crossformer",
        "concepts_n_cmapss/prognostics/lstm",
        "mzvav/diagnostics/lstm",
        "airbus_helicopter/anomaly_detection/isolation_forest",
        "threew/anomaly_detection/isolation_forest",
    ],
)
def test_real_experiment_processed_split_dict_keeps_unit_lists(
    experiment_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify run.py handoff keeps list-per-unit payloads.

    Parameters
    ----------
    experiment_name : str
        Experiment config path to compose and run through preprocessing.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to set the project root for Hydra path resolution.
    """
    monkeypatch.setenv("PROJECT_ROOT", str(PROJECT_ROOT))

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="run.yaml",
            overrides=[
                f"experiment={experiment_name}",
                "hydra/hydra_logging=default",
                "hydra/job_logging=default",
            ],
        )

    try:
        datasource = instantiate(cfg.datasource)
    except InstantiationException as exc:
        pytest.skip(f"Datasource dependencies unavailable for {experiment_name}: {exc}")
    try:
        datasource.load_data()
    except (FileNotFoundError, OSError, ImportError) as exc:
        pytest.skip(f"Dataset unavailable for {experiment_name}: {exc}")

    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    preprocessor = PreProcessor(datasource=datasource, transforms=transforms_manager)
    preprocessor.pipeline(cache_preprocessed=False)

    processed_split = preprocessor.get_processed_split_dict(
        view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
    )
    input_keys = list(cfg.task_definition.model.data_requirements.input_tensors)

    for split in ("train", "val", "test"):
        assert split in processed_split
        for key in input_keys:
            assert (
                key in processed_split[split]
            ), f"Missing key {key!r} in split {split!r} for {experiment_name}."
            assert isinstance(processed_split[split][key], list), (
                f"{experiment_name} returned {type(processed_split[split][key]).__name__} "
                f"for {split}/{key}; run.py expects list-per-unit payloads."
            )
