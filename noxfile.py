from datetime import datetime, timezone
import json
from pathlib import Path
import nox
from itertools import product
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table


# For later with UV:
# import nox

# PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]

# nox.options.default_venv_backend = "uv"


# @nox.session(python=PYTHON_VERSIONS)
# def tests(session):
#     session.run_install(
#         "uv",
#         "sync",
#         env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
#     )
#     session.run(
#         "pytest",
#         "--cov-report=term-missing",
#         "--cov-fail-under=0",
#         *session.posargs,
#     )


@dataclass(frozen=True)
class Info:
    ds: str
    subexp: str
    type: str


DEEP_LEARNING_MODELS = [
    "stf",
    "crossformer",
    "timeseries_transformer",
    "tide",
    "patchtst",
    "lstm",
    "cnn_1d",
    "mlp",
]


TABULAR_STYLE_MODELS = [
    "tabpfn_fit_predict",
    "tabdpt_fit_predict",
    "xgboost_fit_predict",
]

PROGNOSTICS_MODELS = (
    DEEP_LEARNING_MODELS
    + [
        "linear_regression",
        "exponential_regression",
    ]
    + TABULAR_STYLE_MODELS
)
DIAGNOSTICS_MODELS = DEEP_LEARNING_MODELS + ["linear_classifier"] + TABULAR_STYLE_MODELS

# additional remarks
# - we select 2 "raw" datasets: one bearing and one battery.
# - phme20 is also "raw", so we can say it works for smaller datasets.

# For final publication
# - TSFM and PFN finetuning is done on top of these datasets
# - test missing values and irregular sampling


# Excluded for now
# Info(ds="nb1", subexp="", type="prognostics"),
# Info(ds="unibo", subexp="raw", type="prognostics"),
# Info(ds="xjtu", subexp="raw", type="prognostics"),


DATASETS_PROGNOSTICS = [
    # We want to test raw for nb14 to show that it is basically impossible
    # to fine tune in reasonable time because one epoch takes around 2-5 hours.
    # Info(ds="nb14", subexp="raw", type="prognostics"),
    # Info(ds="nb14", subexp="combined", type="prognostics"),
    # Info(ds="unibo", subexp="combined", type="prognostics"),
    # Info(ds="phme20", subexp="raw", type="prognostics"),
    # Info(ds="concepts_n_cmapss", subexp="", type="prognostics"),
    # Info(ds="concepts_n_cmapss_ds02", subexp="", type="prognostics"),
    # # We want to test raw for pronostia to show that it is basically impossible
    # # to fine tune in reasonable time because one epoch takes around 2-5 hours.
    # # Info(ds="pronostia", subexp="raw", type="prognostics"),
    # Info(ds="pronostia", subexp="in_domain/combined", type="prognostics"),
    # Info(ds="xjtu_sy", subexp="in_domain/combined", type="prognostics"),
    Info(ds="xjtu_sy", subexp="phmd_split/combined", type="prognostics"),
]

DATASETS_DIAGNOSTICS = [
    Info(ds="mzvav", subexp="", type="diagnostics"),
    Info(ds="hsf15/accumulator", subexp="", type="diagnostics"),  # hardest task
    Info(ds="hsf15/cooler", subexp="", type="diagnostics"),
    Info(ds="hsf15/pump", subexp="", type="diagnostics"),
    Info(ds="hsf15/valve", subexp="", type="diagnostics"),
    Info(ds="concepts_n_cmapss_multi", subexp="", type="diagnostics"),
]


DATASETS_ANOMALY_DETECTION = [
    Info(ds="airbus_helicopter", subexp="", type="anomaly_detection"),
]

ALL_MODELS = DEEP_LEARNING_MODELS
VENV_PYTHON = Path(".venv/bin/python")
RESULTS_FILE = Path(
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}_experiment_results.jsonl"
)

ANOMALY_DETECTION_MODELS = ["isolation_forest"]

DEEP_LEARNING_ARGS = [
    "debug=nox_testing",
    "num_threads=1",
    "seed=50",
    "trainer.max_epochs=1",
]

TABULAR_STYLE_ARGS = [
    "debug=nox_testing_fit_predict",
    "num_threads=1",
    "seed=50",
    "task_definition.seq_len=10",
]


def run_experiment(session, exp_name: str, metadata: dict, max_err_len: int = 300):
    out = Path(session.create_tmp()) / "result.json"

    status = "success"
    error = None
    model_name = metadata.get("model_name", "")
    if model_name in TABULAR_STYLE_MODELS:
        additional_args = TABULAR_STYLE_ARGS
    else:
        additional_args = DEEP_LEARNING_ARGS

    try:
        session.env.update(
            {
                "HYDRA_FULL_ERROR": "1",
                "PYTHONFAULTHANDLER": "1",
                "NOX_RUN_DETAILS_JSON": str(out),
            }
        )

        session.run(
            str(VENV_PYTHON),
            "picid/run.py",
            "paths=rt_local",
            f"experiment={exp_name}",
            *additional_args,
        )

        run_data = json.loads(out.read_text(encoding="utf-8"))
        session.log("GOT RESULTS: %s", run_data)

    except nox.command.CommandFailed as e:
        status = "failed"
        error = str(e).replace("\n", " ")[:max_err_len]
        raise
    finally:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "experiment": exp_name,
            **metadata,
            **(run_data if status == "success" else {}),
            "error": error,
        }
        with RESULTS_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")


@nox.session(venv_backend="none")
@nox.parametrize("model_name,info", product(DIAGNOSTICS_MODELS, DATASETS_DIAGNOSTICS))
def experiments_diagnostics(session, model_name, info):
    dataset_key = info.ds
    subexp = info.subexp
    task_type = info.type

    if subexp is not None and subexp != "":
        exp_name = f"{dataset_key}/{task_type}/{subexp}/{model_name}"
    else:
        exp_name = f"{dataset_key}/{task_type}/{model_name}"

    run_experiment(
        session,
        exp_name,
        metadata={"model_name": model_name, "dataset_key": dataset_key},
    )


@nox.session(venv_backend="none")
@nox.parametrize("model_name,info", product(PROGNOSTICS_MODELS, DATASETS_PROGNOSTICS))
def experiments_prognostics(session, model_name, info):
    dataset_key = info.ds
    subexp = info.subexp
    task_type = info.type

    if subexp is not None and subexp != "":
        exp_name = f"{dataset_key}/{task_type}/{subexp}/{model_name}"
    else:
        exp_name = f"{dataset_key}/{task_type}/{model_name}"

    run_experiment(
        session,
        exp_name,
        metadata={"model_name": model_name, "dataset_key": dataset_key},
    )


@nox.session(venv_backend="none")
@nox.parametrize(
    "model_name,info", product(ANOMALY_DETECTION_MODELS, DATASETS_ANOMALY_DETECTION)
)
def experiments_anomaly_detection(session, model_name, info):
    dataset_key = info.ds
    subexp = info.subexp
    task_type = info.type

    if subexp is not None and subexp != "":
        exp_name = f"{dataset_key}/{task_type}/{subexp}/{model_name}"
    else:
        exp_name = f"{dataset_key}/{task_type}/{model_name}"

    run_experiment(
        session,
        exp_name,
        metadata={"model_name": model_name, "dataset_key": dataset_key},
    )


@nox.session(venv_backend="none")
def analyze_results(session):
    """Analyze the most recent experiment results and display tensor size table."""

    # Find the most recent results file
    results_files = sorted(Path(".").glob("*_experiment_results.jsonl"), reverse=True)

    if not results_files:
        session.log("No results files found.")
        return

    most_recent = results_files[0]
    session.log(f"Reading results from: {most_recent}")

    # Collect tensor size information
    tensor_data = []

    with most_recent.open("r") as f:
        for line in f:
            record = json.loads(line)

            print(record)

            # Extract torch.Size fields
            torch_sizes = {}
            for key, value in record.items():
                if isinstance(value, str) and value.startswith("torch.Size("):
                    torch_sizes[key] = value

            tensor_data.append(
                {
                    "experiment": record.get("experiment", ""),
                    "model": record.get("model_name", ""),
                    "dataset": record.get("dataset_key", ""),
                    "status": record.get("status", ""),
                    **(torch_sizes if torch_sizes else {}),
                }
            )

    if not tensor_data:
        session.log("No tensor size data found in results.")
        return

    # Create and populate Rich table
    console = Console()
    table = Table(title=f"Experiment Results - {most_recent.name}")

    if tensor_data:
        # Collect all unique headers from all records
        headers = list({key for row in tensor_data for key in row.keys()})

        # Add columns
        for header in headers:
            table.add_column(header, style="cyan", no_wrap=False)

        # Add rows
        for row in tensor_data:
            status = row.get("status", "")
            style = "red" if status == "failed" else None
            table.add_row(*[str(row.get(h, "")) for h in headers], style=style)

        console.print(table)
