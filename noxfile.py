from datetime import datetime, timezone
import json
from pathlib import Path
import nox
from itertools import product
from dataclasses import dataclass


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
    # "stf",
    # "crossformer",
    # "timeseries_transformer",
    # "tide",
    # "patchtst",
    "lstm",
    # "cnn_1d",
    # "exponential_regression",
    # "linear_regression",
    "mlp",
]
TABULAR_STYLE_MODELS = [
    # "tabpfn_fit_predict",
    # "tabdpt_fit_predict",
    # "xgboost_fit_predict",
]

# additional remarks
# - we select 2 "raw" datasets: one bearing and one battery.
# - phme20 is also "raw", so we can say it works for smaller datasets.

# For final publication
# - TSFM and PFN finetuning is done on top of these datasets
# - test missing values and irregular sampling


DATASETS_PROGNOSTICS = [
    # Info(ds="nb1", subexp="", type="prognostics"),
    # We want to test raw for nb14 to show that it is basically impossible
    # to fine tune in reasonable time because one epoch takes around 2-5 hours.
    Info(ds="nb14", subexp="raw", type="prognostics"),
    Info(ds="nb14", subexp="statistics", type="prognostics"),
    # Info(ds="unibo", subexp="raw", type="prognostics"),
    Info(ds="unibo", subexp="statistics", type="prognostics"),
    Info(ds="phme20", subexp="raw", type="prognostics"),
    Info(ds="concepts_n_cmapss", subexp="", type="prognostics"),
    Info(ds="concepts_n_cmapss_ds02", subexp="", type="prognostics"),
    # We want to test raw for pronostia to show that it is basically impossible
    # to fine tune in reasonable time because one epoch takes around 2-5 hours.
    Info(ds="pronostia", subexp="raw", type="prognostics"),
    Info(ds="pronostia", subexp="statistics", type="prognostics"),
    # Info(ds="xjtu", subexp="raw", type="prognostics"),
    Info(ds="xjtu", subexp="statistics", type="prognostics"),
    # "concepts_n_cmapss",
    # "concepts_n_cmapss_ds02",
]

DATASETS_DIAGNOSTICS = [
    Info(ds="mzvav", subexp="", type="diagnostics"),
    Info(ds="hsf15/accumulator", subexp="", type="diagnostics"),  # hardest task
    Info(ds="hsf15/cooler", subexp="", type="diagnostics"),
    Info(ds="hsf15/pump", subexp="", type="diagnostics"),
    Info(ds="hsf15/valve", subexp="", type="diagnostics"),
    Info(ds="concepts_n_cmapss_multi", subexp="", type="diagnostics"),
]


ALL_MODELS = DEEP_LEARNING_MODELS + TABULAR_STYLE_MODELS
VENV_PYTHON = Path(".venv/bin/python")
RESULTS_FILE = Path(
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}_experiment_results.jsonl"
)

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
]


def run_experiment(session, exp_name: str, metadata: dict, max_err_len: int = 300):
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
            }
        )

        session.run(
            str(VENV_PYTHON),
            "lmetk/run.py",
            "paths=rt_local",
            f"experiment={exp_name}",
            *additional_args,
        )
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
            "error": error,
        }
        with RESULTS_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")


@nox.session(venv_backend="none")
@nox.parametrize("model_name,info", product(ALL_MODELS, DATASETS_DIAGNOSTICS))
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
@nox.parametrize("model_name,info", product(ALL_MODELS, DATASETS_PROGNOSTICS))
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
