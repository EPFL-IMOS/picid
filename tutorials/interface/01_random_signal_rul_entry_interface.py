"""
EntryInterface: random signal -> random RUL signal.

A minimal end-to-end exercise of the picid `EntryInterface` pipeline against a
purely synthetic dataset. Both the input features and the regression target
(`rul`) are drawn from a Gaussian distribution, so the resulting metrics only
prove the wiring works — they say nothing about a real degradation pattern.

The script walks through the same steps as
`01_random_signal_rul_entry_interface.ipynb`:

    1. Synthesize a small DataFrame with N_FEATURES random signals + a `rul`
       column.
    2. Wrap it in a `CustomSingleSourceLoader` and split it via `TimeSplitter`.
    3. Build two transforms programmatically and pre-process the datasource.
    4. Run a structural split-alignment report and visualise it.
    5. Train a tiny MLP and report test metrics.

Run from the repository root so picid's config_path resolves correctly:

    python tutorials/interface/01_random_signal_rul_entry_interface.py
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from picid.data.data_objects.validation import (
    build_split_alignment_report_table,
    collect_split_alignment_report,
)
from picid.data.preprocessing import TimeSplitter
from picid.interface import CustomSingleSourceLoader, EntryInterface
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.task_definition import Prognostic
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn
from picid.transforms.base_transforms.stfft import STFTTransform

# Quiet the noisier picid loggers so the tutorial output stays readable.
logging.getLogger("picid.interface.interface").setLevel(logging.WARNING)
logging.getLogger("picid.interface.datasources").setLevel(logging.WARNING)
logging.getLogger("picid.data.datasets.hydra_concat_dataset").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------
# `CustomSingleSourceLoader.load_from_csv` expects the regression target to be
# one column in the table, so `rul` is appended as the last column.

SEED = 7
SEQ_LEN = 12
N_STEPS = 240
N_FEATURES = 3

rng = np.random.default_rng(SEED)
features = rng.normal(size=(N_STEPS, N_FEATURES)).astype(np.float32)
rul = rng.normal(size=(N_STEPS, 1)).astype(np.float32)

columns = [f"signal_{i}" for i in range(N_FEATURES)] + ["rul"]
frame = pd.DataFrame(np.concatenate([features, rul], axis=1), columns=columns)
print(frame.head())


# ---------------------------------------------------------------------------
# 2. Datasource + splitter
# ---------------------------------------------------------------------------
# `TimeSplitter` carves the full sequence into train/val/test chunks along the
# time axis. `seq_len` matches the prognostics task definition declared below;
# `pred_len=0` is the RUL-style "predict the current label" setting.

splitter = TimeSplitter(
    train=0.6,
    val=0.2,
    test=None,
    seq_len=SEQ_LEN,
    pred_len=0,
    create_splits_for=["features", "timestamps", "rul"],
)

datasource = CustomSingleSourceLoader.load_from_csv(
    source=frame,
    target_column="rul",
    task_mode="rul",
    data_splitter=splitter,
    data_name="random_signal_rul",
)


# ---------------------------------------------------------------------------
# 3. Transforms (programmatic)
# ---------------------------------------------------------------------------
# Each transform is a `DataTransform` wrapper around a base transform plus
# metadata describing where it reads from / writes to.
#
# - `scaler_features` fits a sklearn MinMaxScaler on the train split and
#   rescales the `features` array in place.
# - `stft_features` runs an STFT over `features` and assigns the spectrogram to
#   a new `stft_features` key. Using `assign_to` (instead of overwriting
#   `features`) keeps the time-domain shape intact for the MLP downstream.

scaler_features = DataTransform(
    transform_name="scaler_features",
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "features", "fit_on": "train"},
)

stft_features = DataTransform(
    transform_name="stft_features",
    transform=STFTTransform(win_len=16, hop=8, output_format="magnitude"),
    metadata={"apply_to": "features", "assign_to": "stft_features"},
)

transforms = [scaler_features, stft_features]

# `process_datasource(..., cache=True)` (the default) memoizes the transformed
# dataset on disk under `project_config.cache_path` (`~/picid/cache/...`). On
# subsequent runs with the same `(datasource, transforms)` pair joblib loads
# the result instead of re-fitting the transforms.
#
# Set `cache=False` while iterating on transform code so changes take effect
# immediately, or to avoid filling the cache with throwaway one-off runs. To
# wipe an existing cache from the shell:
#     rm -rf ~/picid/cache/picid/interface/utils/InterfacePreProcessor
interface = EntryInterface()
processed = interface.process_datasource(datasource, transforms=transforms, cache=False)

# `processed.data_dict` is a nested {split: {key: [unit_arrays]}} structure.
# Print a shapes-only view as a sanity check.
shapes = {
    split: {
        key: [arr.shape for arr in values]
        for key, values in processed.data_dict[split].items()
    }
    for split in ["train", "val", "test"]
}
print(shapes)


# ---------------------------------------------------------------------------
# 4. Split alignment report
# ---------------------------------------------------------------------------
# `collect_split_alignment_report` walks every (key, split) pair and reports
# unit counts, sample shapes, and whether each split's payloads share the same
# schema. The visualisation pairs that with two matplotlib panels:
#
# - left:  bar chart of unit counts per split for each data key,
# - right: red/yellow/green grid colouring each (key, split) cell by
#          `schema_status` (`homogeneous`, `heterogeneous`, or `empty`).

data_dict = processed.data_dict
keys = sorted({k for split_data in data_dict.values() for k in split_data.keys()})
payloads = [
    (key, {split: data_dict[split].get(key, []) for split in data_dict}) for key in keys
]

report = collect_split_alignment_report(payloads)
Console(width=180).print(build_split_alignment_report_table(report))

splits = report["splits"]
report_keys = [row["key"] for row in report["rows"]]

# Map status -> colour value: homogeneous=1 (green), empty=0 (yellow),
# heterogeneous=-1 (red). Anything else falls back to 0.
status_to_int = {"empty": 0, "homogeneous": 1, "heterogeneous": -1}

fig, axes = plt.subplots(1, 2, figsize=(11, 0.4 * len(report_keys) + 2.5))

x = np.arange(len(report_keys))
width = 0.8 / max(len(splits), 1)
for i, split in enumerate(splits):
    counts = [(row["counts"][split] or 0) for row in report["rows"]]
    offset = (i - (len(splits) - 1) / 2) * width
    axes[0].bar(x + offset, counts, width, label=split)
axes[0].set_xticks(x)
axes[0].set_xticklabels(report_keys, rotation=45, ha="right")
axes[0].set_ylabel("unit count")
axes[0].set_title("Units per split")
axes[0].legend()

status_matrix = np.array(
    [[status_to_int[row["schema_status"][s]] for s in splits] for row in report["rows"]]
)
axes[1].imshow(status_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
axes[1].set_xticks(range(len(splits)))
axes[1].set_xticklabels(splits)
axes[1].set_yticks(range(len(report_keys)))
axes[1].set_yticklabels(report_keys)
for i, row in enumerate(report["rows"]):
    for j, s in enumerate(splits):
        axes[1].text(
            j, i, row["schema_status"][s], ha="center", va="center", fontsize=8
        )
axes[1].set_title("Schema status")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 5. Train + test
# ---------------------------------------------------------------------------
# `Prognostic` declares the task shape (RUL regression with a sliding window of
# length SEQ_LEN). Passing the already-`processed` datasource skips the inline
# preprocessing step inside `train()`. `transforms=[]` here is intentional —
# the transforms have already been applied during `process_datasource`.

task_definition = Prognostic(
    task_type="rul",
    seq_len=SEQ_LEN,
    stride=4,
    stride_train=4,
)

# Pass the logger as a `train()` attribute (not a Hydra group swap). When
# `loggers=` is non-empty, `train()` skips the YAML logger defaults, so this
# CSVLogger fully replaces the wandb default declared in `configs/run.yaml`.
# `datamodule.num_workers=0` is the only override left: multi-worker fork is
# fine inside Jupyter but flaky when this file is invoked as a plain script.
results = interface.train(
    run_name="entry_interface_random_signal_rul",
    model="mlp",
    task_definition=task_definition,
    datasource=processed,
    transforms=[],
    evaluators="default",
    loggers=[CsvLogger(name="entry_interface_random_signal_rul")],
    overrides=["datamodule.num_workers=0"],
    enable_progress_bar=False,
    seed=SEED,
)

# `results` is the standard Lightning test-result list. Because both inputs and
# targets are random, these numbers only confirm the pipeline ran end-to-end.
print(results)
