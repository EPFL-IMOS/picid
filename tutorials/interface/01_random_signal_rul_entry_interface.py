"""EntryInterface: manually composed ragged units and two model families.

This tutorial exercises the picid ``EntryInterface`` pipeline with a small,
fully synthetic RUL dataset. The signals contain a weak degradation trend so
the example is realistic enough to inspect, but the resulting metrics are only
intended to prove that the wiring works.

The script demonstrates how to:

1. Create several named units with different time-series lengths.
2. Compose ``CustomSingleSourceLoader`` objects into one datasource by hand.
3. Recognise ragged data and inspect its per-unit shapes without padding.
4. Preprocess the datasource into model-specific views.
5. Compare a fit/predict gradient-boosted model with a deep-learning MLP.
6. Train and evaluate both models through the same ``EntryInterface`` API.

Run from the repository root so picid's config path resolves correctly:

    uv run python tutorials/interface/01_random_signal_rul_entry_interface.py
"""

from copy import deepcopy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from picid.data.data_objects.validation import (
    build_split_alignment_report_table,
    collect_split_alignment_report,
)
from picid.data.preprocessing import TimeSplitter
from picid.interface import (
    CustomMultiSourceLoader,
    CustomSingleSourceLoader,
    EntryInterface,
)
from picid.interface.schemas.loggers import CsvLogger
from picid.interface.schemas.task_definition import Prognostic
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.reshaping import ReshapeTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn

# Quiet the noisier picid loggers so the tutorial output stays readable.
logging.getLogger("picid.interface.interface").setLevel(logging.WARNING)
logging.getLogger("picid.interface.datasources").setLevel(logging.WARNING)
logging.getLogger("picid.data.datasets.hydra_concat_dataset").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 1. Create several synthetic units
# ---------------------------------------------------------------------------
# Each DataFrame represents one physical unit. The units have the same columns
# but different numbers of time steps, which is common for run-to-failure data.

SEED = 7
SEQ_LEN = 12
N_FEATURES = 3
UNIT_LENGTHS = {
    "unit_short": 180,
    "unit_medium": 240,
    "unit_long": 300,
}


def make_unit_frame(
    *, unit_index: int, n_steps: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Create one synthetic degradation trajectory and its RUL target."""

    progress = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    health = 1.0 - progress
    operating_offset = 0.15 * unit_index

    features = np.column_stack(
        [
            health + operating_offset + rng.normal(0.0, 0.05, n_steps),
            np.sin(progress * 2.0 * np.pi) + rng.normal(0.0, 0.05, n_steps),
            progress**2 + rng.normal(0.0, 0.03, n_steps),
        ]
    ).astype(np.float32)
    rul = health.reshape(-1, 1)

    columns = [f"signal_{i}" for i in range(N_FEATURES)] + ["rul"]
    return pd.DataFrame(np.concatenate([features, rul], axis=1), columns=columns)


rng = np.random.default_rng(SEED)
unit_frames = {
    unit_name: make_unit_frame(
        unit_index=unit_index,
        n_steps=n_steps,
        rng=rng,
    )
    for unit_index, (unit_name, n_steps) in enumerate(UNIT_LENGTHS.items())
}

unit_summary = pd.DataFrame(
    [
        {
            "unit": unit_name,
            "time_steps": len(frame),
            "feature_columns": N_FEATURES,
        }
        for unit_name, frame in unit_frames.items()
    ]
)
print("\nRaw units:\n", unit_summary.to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Compose the units by hand
# ---------------------------------------------------------------------------
# A TimeSplitter partitions every unit independently along its time axis. Each
# child receives its own copy because splitters carry configuration/state.

splitter_template = TimeSplitter(
    train=0.6,
    val=0.2,
    test=None,
    seq_len=SEQ_LEN,
    pred_len=0,
    create_splits_for=["features", "timestamps", "rul"],
)

unit_loaders = {
    unit_name: CustomSingleSourceLoader.load_from_csv(
        source=frame,
        target_column="rul",
        task_mode="rul",
        data_splitter=deepcopy(splitter_template),
        data_name=unit_name,
    )
    for unit_name, frame in unit_frames.items()
}

# The dictionary keys are the human-readable unit identities. Passing loader
# instances explicitly is the instance-driven composition mode; alternatively,
# CustomMultiSourceLoader.load_from_primitive(...) can create the children.
datasource = CustomMultiSourceLoader(
    sources=unit_loaders,
    task_mode="rul",
    data_name="manual_ragged_units",
)


# ---------------------------------------------------------------------------
# 3. Preprocess and explain ragged data
# ---------------------------------------------------------------------------
# Return fresh transform objects for each preprocessing run: fitted scaler
# state must not be shared between model-specific views.


def scaling_transforms() -> list[DataTransform]:
    """Build common feature and target scalers for one preprocessing run."""

    return [
        DataTransform(
            transform_name="scale_features",
            transform=MinMaxScalerSklearn(),
            metadata={"apply_to": "features", "fit_on": "train"},
        ),
        DataTransform(
            transform_name="scale_rul",
            transform=MinMaxScalerSklearn(),
            metadata={"apply_to": "rul", "fit_on": "train"},
        ),
    ]


interface = EntryInterface()

# The deep-learning view retains a list of 2D arrays per split. The arrays have
# different time lengths but agree on feature width; no padding or truncation
# is needed because the downstream dataset windows each unit independently.
deep_learning_processed = interface.process_datasource(
    datasource,
    transforms=scaling_transforms(),
    cache=False,
)

unit_names = list(unit_loaders)
unit_lengths_by_split = {
    split: {
        unit_name: len(unit_features)
        for unit_name, unit_features in zip(
            unit_names,
            split_data["features"],
            strict=True,
        )
    }
    for split, split_data in deep_learning_processed.data_dict.items()
}

print("\nPer-unit lengths after splitting:")
for split in ("train", "val", "test"):
    lengths = unit_lengths_by_split[split]
    is_ragged = len(set(lengths.values())) > 1
    print(f"  {split:>5}: {lengths} -> ragged={is_ragged}")

print(
    "\nRagged means that units have different lengths along the time axis. "
    "picid stores them as list[np.ndarray] instead of forcing np.stack(), "
    "padding, or truncation."
)

# Fit/predict datasets consume complete tabular tasks rather than sliding
# windows. Add a leading task dimension to every unit:
#
#     deep learning: (time, features)
#     fit/predict:    (1, time, features)
#
# The fit/predict dataset then concatenates the variable-length units into one
# tabular task. This preserves every sample while avoiding manual padding.
fit_predict_processed = interface.process_datasource(
    datasource,
    transforms=[
        *scaling_transforms(),
        DataTransform(
            transform_name="add_task_dim_features",
            transform=ReshapeTransform("time feature -> 1 time feature"),
            metadata={"apply_to": "features", "assign_to": "features"},
        ),
        DataTransform(
            transform_name="add_task_dim_rul",
            transform=ReshapeTransform("time target -> 1 time target"),
            metadata={"apply_to": "rul", "assign_to": "rul"},
        ),
    ],
    cache=False,
)

deep_shapes = [
    unit.shape for unit in deep_learning_processed.data_dict["train"]["features"]
]
fit_predict_shapes = [
    unit.shape for unit in fit_predict_processed.data_dict["train"]["features"]
]
print(f"\nDeep-learning train-unit shapes: {deep_shapes}")
print(f"Fit/predict train-unit shapes:    {fit_predict_shapes}")


# ---------------------------------------------------------------------------
# 4. Inspect and visualise split alignment
# ---------------------------------------------------------------------------

data_dict = deep_learning_processed.data_dict
keys = sorted({key for split_data in data_dict.values() for key in split_data})
payloads = [
    (key, {split: data_dict[split].get(key, []) for split in data_dict}) for key in keys
]

report = collect_split_alignment_report(payloads)
Console(width=180).print(build_split_alignment_report_table(report))

splits = report["splits"]
report_keys = [row["key"] for row in report["rows"]]
status_to_int = {"empty": 0, "homogeneous": 1, "heterogeneous": -1}

fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * len(report_keys) + 2.5))

# Left: the unequal bar heights make ragged unit lengths visible.
x = np.arange(len(unit_names))
width = 0.8 / max(len(splits), 1)
for split_index, split in enumerate(splits):
    lengths = [unit_lengths_by_split[split][name] for name in unit_names]
    offset = (split_index - (len(splits) - 1) / 2) * width
    axes[0].bar(x + offset, lengths, width, label=split)
axes[0].set_xticks(x)
axes[0].set_xticklabels(unit_names, rotation=25, ha="right")
axes[0].set_ylabel("time steps")
axes[0].set_title("Variable per-unit lengths (ragged data)")
axes[0].legend()

# Right: confirm that every key still has a compatible schema in each split.
status_matrix = np.array(
    [
        [status_to_int[row["schema_status"][split]] for split in splits]
        for row in report["rows"]
    ]
)
axes[1].imshow(status_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
axes[1].set_xticks(range(len(splits)))
axes[1].set_xticklabels(splits)
axes[1].set_yticks(range(len(report_keys)))
axes[1].set_yticklabels(report_keys)
for row_index, row in enumerate(report["rows"]):
    for split_index, split in enumerate(splits):
        axes[1].text(
            split_index,
            row_index,
            row["schema_status"][split],
            ha="center",
            va="center",
            fontsize=8,
        )
axes[1].set_title("Schema status")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 5. Fit/predict versus deep learning
# ---------------------------------------------------------------------------

comparison = Table(title="Two model execution styles")
comparison.add_column("Aspect", style="bold cyan")
comparison.add_column("Fit/predict")
comparison.add_column("Deep learning")
comparison.add_row("Model", "xgboost_fit_predict", "mlp")
comparison.add_row("Training API", "fit(X, y)", "minibatches + backpropagation")
comparison.add_row("Input view", "complete tabular task", "sliding windows per unit")
comparison.add_row("Optimisation", "gradient-boosted trees", "AdamW over epochs")
comparison.add_row(
    "Ragged units", "concatenated without padding", "windowed separately"
)
Console().print(comparison)

task_definition = Prognostic(
    task_type="rul",
    seq_len=SEQ_LEN,
    stride=4,
    stride_train=4,
)

common_train_kwargs = {
    "task_definition": task_definition,
    "transforms": [],  # both datasource views were already processed above
    "evaluators": "default",
    "enable_progress_bar": False,
    "seed": SEED,
}

# Fit/predict models do not use gradient descent. The Lightning trainer is only
# an orchestration shell around fit() and predict(), so max_epochs must stay 1.
fit_predict_results = interface.train(
    run_name="entry_interface_fit_predict",
    model="xgboost_fit_predict",
    datasource=fit_predict_processed,
    loggers=[CsvLogger(name="entry_interface_fit_predict")],
    overrides=[
        "trainer.max_epochs=1",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "datamodule.num_workers=0",
        "+model.n_estimators=25",
    ],
    **common_train_kwargs,
)

# The MLP consumes seq_len-sized windows and updates its weights with
# backpropagation. One epoch keeps the tutorial fast; use more for real work.
deep_learning_results = interface.train(
    run_name="entry_interface_deep_learning",
    model="mlp",
    datasource=deep_learning_processed,
    loggers=[CsvLogger(name="entry_interface_deep_learning")],
    overrides=[
        "trainer.max_epochs=1",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "datamodule.num_workers=0",
    ],
    **common_train_kwargs,
)


# ---------------------------------------------------------------------------
# 6. Compare returned metrics
# ---------------------------------------------------------------------------
# Both calls return the standard Lightning list of test metric dictionaries.
# The target was scaled, so metric names include the ``_normalized`` suffix.

result_rows = []
for model_family, model_name, results in [
    ("fit/predict", "xgboost_fit_predict", fit_predict_results),
    ("deep learning", "mlp", deep_learning_results),
]:
    metrics = results[0] if results else {}
    result_rows.append(
        {
            "family": model_family,
            "model": model_name,
            **metrics,
        }
    )

print("\nReturned test metrics:")
print(pd.DataFrame(result_rows).to_string(index=False))
print(
    "\nThese one-epoch numbers compare execution paths, not model quality. "
    "Increase epochs/estimators and use meaningful data for an experiment."
)
