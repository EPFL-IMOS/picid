"""
Fixtures for picid_report tests: mock W&B-style DataFrames.

Provides synthetic experiment data with:
- Heterogeneous HP types (int vs float) for string-normalization merge testing
- Multiple runs per config for mean/std/(n=count) validation
- Optional NaN in performance columns for Numeric Guard logic
- Custom prefixes (val_best_rerun/) for metric capture vs HP exclusion
"""

import numpy as np
import pandas as pd
import pytest

from picid_report import config


# --- Column names from config (do not modify picid_report, only read) ---
MODEL_COL = config.COLUMN_CONFIG["model_target"]
DATASET_COL = config.COLUMN_CONFIG["dataset_name"]
OPT_METRIC_COL = config.COLUMN_CONFIG["optimization_metric"]
OPT_MODE_COL = config.COLUMN_CONFIG["optimization_mode"]
TARGET_METRIC_COL = config.COLUMN_CONFIG["target_metric"]
TARGET_MODE_COL = config.COLUMN_CONFIG["target_metric_mode"]
EVALUATOR_METRICS_COL = config.COLUMN_CONFIG["evaluator_metrics"]
DATA_SEED_COL = config.DEFAULT_DATA_SEED_COL
MODEL_SEED_COL = config.DEFAULT_MODEL_SEED_COL


@pytest.fixture
def base_config_columns():
    """Columns that exist after run_processor normalization (config-derived)."""
    return [
        MODEL_COL,
        DATASET_COL,
        MODEL_SEED_COL,
        "run_name",
        OPT_METRIC_COL,
        OPT_MODE_COL,
        "task_definition.seq_len",
        "optimization.lr",
    ]


@pytest.fixture
def mock_df_heterogeneous_hp(base_config_columns):
    """
    Mock W&B DataFrame with heterogeneous hyperparameter types (int vs float).

    Methodology: Same logical value appears as 150 (int) in one run and 150.0 (float)
    in another to trigger merge key mismatch unless string normalization is applied.
    Used to validate the left-join logic for grid completion in schema-first mode.
    """
    rows = [
        {
            MODEL_COL: "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 1,
            "run_name": "run_a",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 150,  # int
            "optimization.lr": 0.001,
            "test/mse": 0.5,
            "val/mse": 0.52,
        },
        {
            MODEL_COL: "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 2,
            "run_name": "run_b",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 150.0,  # float - same value, different type
            "optimization.lr": 0.001,
            "test/mse": 0.48,
            "val/mse": 0.51,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mock_df_multi_seed_runs(base_config_columns):
    """
    Mock DataFrame with multiple runs per configuration for mean, std, (n=count).

    Methodology: Two configs (seq_len=10 and seq_len=20), three seeds each, so
    aggregation yields mean ± std and (n=3) for each. Validates aggregate_and_find_best
    and reporting format 'mean ± std (n=count)'.
    """
    data = []
    for seq_len in (10, 20):
        for seed, mse in [(1, 0.4), (2, 0.5), (3, 0.6)]:
            data.append(
                {
                    MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
                    DATASET_COL: "DS1",
                    MODEL_SEED_COL: seed,
                    "run_name": f"run_{seq_len}_{seed}",
                    OPT_METRIC_COL: "test/mse",
                    OPT_MODE_COL: "min",
                    "task_definition.seq_len": seq_len,
                    "optimization.lr": 0.001,
                    "test/mse": mse,
                    "val/mse": mse + 0.02,
                }
            )
    return pd.DataFrame(data)


@pytest.fixture
def mock_df_with_nan_metrics(base_config_columns):
    """
    Mock DataFrame with NaN in performance columns to test Numeric Guard logic.

    Methodology: One run has NaN for test/mse; aggregate_and_find_best coerces to
    numeric and groupby still produces mean/std/count (NaN propagates or is skipped).
    Validates pd.to_numeric(..., errors='coerce') and non-crash behavior.
    """
    rows = [
        {
            MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 1,
            "run_name": "run_ok",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 10,
            "optimization.lr": 0.001,
            "test/mse": 0.4,
            "val/mse": 0.42,
        },
        {
            MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 2,
            "run_name": "run_nan",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 10,
            "optimization.lr": 0.001,
            "test/mse": np.nan,
            "val/mse": np.nan,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mock_df_val_best_rerun_prefix(base_config_columns):
    """
    Mock DataFrame with val_best_rerun/ metrics to ensure they are captured as metrics
    but not treated as hyperparameters (COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH).
    """
    rows = [
        {
            MODEL_COL: "baselines.patchtst_model.PatchTST_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 1,
            "run_name": "run_1",
            OPT_METRIC_COL: "val_best_rerun/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 10,
            "optimization.lr": 0.001,
            "test/mse": 0.3,
            "val/mse": 0.32,
            "val_best_rerun/mse": 0.28,
        },
        {
            MODEL_COL: "baselines.patchtst_model.PatchTST_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 2,
            "run_name": "run_2",
            OPT_METRIC_COL: "val_best_rerun/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 10,
            "optimization.lr": 0.001,
            "test/mse": 0.31,
            "val/mse": 0.33,
            "val_best_rerun/mse": 0.29,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mock_df_partial_sweep(base_config_columns):
    """
    Partial sweep: only 2 of 6 grid points completed. Used for schema-first test:
    when EXPECTED_SEARCH_SPACE is provided, result must contain all grid entries,
    with missing runs shown as - (NaN in metrics).
    """
    # Grid: seq_len in [10, 50], lr in [0.001, 0.0005, 0.0001] -> 6 points.
    # We only provide runs for (10, 0.001) and (50, 0.0005).
    rows = [
        {
            MODEL_COL: "baselines.tide_model.TiDE_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 1,
            "run_name": "r1",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 10,
            "optimization.lr": 0.001,
            "test/mse": 0.2,
            "val/mse": 0.22,
        },
        {
            MODEL_COL: "baselines.tide_model.TiDE_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 2,
            "run_name": "r2",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 50,
            "optimization.lr": 0.0005,
            "test/mse": 0.25,
            "val/mse": 0.27,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mock_df_single_run_n1(base_config_columns):
    """
    Single run per config (n=1) to verify std defaults to 0.0 and formatting does not crash.
    """
    rows = [
        {
            MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 1,
            "run_name": "only_run",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 10,
            "optimization.lr": 0.001,
            "test/mse": 0.44,
            "val/mse": 0.46,
        }
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mock_df_two_model_wrappers(base_config_columns):
    """
    Two StatisticalBaselineWrapper variants (linear vs exponential) to verify
    model_type differentiation during preprocessing (subset by model_target).
    """
    rows = []
    for model in [
        "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)",
        "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)",
    ]:
        rows.append(
            {
                MODEL_COL: model,
                DATASET_COL: "DS1",
                MODEL_SEED_COL: 1,
                "run_name": f"run_{model.split()[-1]}",
                OPT_METRIC_COL: "test/mse",
                OPT_MODE_COL: "min",
                "task_definition.seq_len": 10,
                "optimization.lr": 0.001,
                "test/mse": 0.1 if "linear" in model else 0.15,
                "val/mse": 0.12 if "linear" in model else 0.17,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def mock_df_no_matching_metrics(base_config_columns):
    """
    DataFrame where no column matches the configured metric_prefixes (e.g. only
    'custom/my_metric' while prefixes are ['val/', 'test/']). Tests zero-metric-match
    path: pipeline must not crash when agg_dict is empty.
    """
    rows = [
        {
            MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 1,
            "run_name": "run_1",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 10,
            "optimization.lr": 0.001,
            "custom/my_metric": 1.0,  # not val/ or test/
        }
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mock_df_partial_seed_config(base_config_columns):
    """
    Three HP configs; one config has only 2 of 3 required model seeds (72, 88, 101).
    Used to test total_runs and configs_failed_not_full_seed_set.
    - Config (seq_len=10, lr=0.001): seeds 72, 88, 101 -> kept (3 runs).
    - Config (seq_len=20, lr=0.001): seeds 72, 88, 101 -> kept (3 runs).
    - Config (seq_len=30, lr=0.001): seeds 72, 88 only -> dropped (not full seed set).
    After model-seed filter: 2 configs, 6 runs, configs_failed = 1.
    """
    data = []
    for seq_len, seeds in [(10, (72, 88, 101)), (20, (72, 88, 101)), (30, (72, 88))]:
        for seed in seeds:
            data.append(
                {
                    MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
                    DATASET_COL: "DS1",
                    MODEL_SEED_COL: seed,
                    "run_name": f"run_{seq_len}_{seed}",
                    OPT_METRIC_COL: "test/mse",
                    OPT_MODE_COL: "min",
                    "task_definition.seq_len": seq_len,
                    "optimization.lr": 0.001,
                    "test/mse": 0.4 + seq_len / 100,
                    "val/mse": 0.42 + seq_len / 100,
                }
            )
    return pd.DataFrame(data)


@pytest.fixture
def mock_df_partial_valid_metric(base_config_columns):
    """
    Two HP configs, both with all required model seeds (72, 88, 101).
    One config (seq_len=20) has one run with NaN for test/mse.
    Used to test configs_failed_missing_invalid_metric: after metric filter
    that config has only 2 valid runs -> dropped -> configs_failed_missing_invalid_metric=1.
    """
    data = []
    # Config seq_len=10: all three seeds with valid test/mse
    for seed in (72, 88, 101):
        data.append(
            {
                MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
                DATASET_COL: "DS1",
                MODEL_SEED_COL: seed,
                "run_name": f"run_10_{seed}",
                OPT_METRIC_COL: "test/mse",
                OPT_MODE_COL: "min",
                "task_definition.seq_len": 10,
                "optimization.lr": 0.001,
                "test/mse": 0.4,
                "val/mse": 0.42,
            }
        )
    # Config seq_len=20: seeds 72 (valid), 88 (NaN), 101 (valid)
    data.append(
        {
            MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 72,
            "run_name": "run_20_72",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 20,
            "optimization.lr": 0.001,
            "test/mse": 0.5,
            "val/mse": 0.52,
        }
    )
    data.append(
        {
            MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 88,
            "run_name": "run_20_88",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 20,
            "optimization.lr": 0.001,
            "test/mse": np.nan,
            "val/mse": np.nan,
        }
    )
    data.append(
        {
            MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 101,
            "run_name": "run_20_101",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 20,
            "optimization.lr": 0.001,
            "test/mse": 0.51,
            "val/mse": 0.53,
        }
    )
    return pd.DataFrame(data)


@pytest.fixture
def mock_df_original_metric_missing_has_fallback(base_config_columns):
    """
    One model with val/mse and test/mse (valid) but no val/loss column.
    Used to test sort-metric fallback: when optimization_col=val/loss would drop all rows,
    pipeline should use fallback (val/mse preferred over test/mse) and keep the model.
    """
    data = [
        {
            MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 72,
            "run_name": "run_10_72",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 10,
            "optimization.lr": 0.001,
            "val/mse": 0.42,
            "test/mse": 0.40,
        },
        {
            MODEL_COL: "baselines.lstm_model.LSTM_Forecaster",
            DATASET_COL: "DS1",
            MODEL_SEED_COL: 88,
            "run_name": "run_10_88",
            OPT_METRIC_COL: "test/mse",
            OPT_MODE_COL: "min",
            "task_definition.seq_len": 10,
            "optimization.lr": 0.001,
            "val/mse": 0.43,
            "test/mse": 0.41,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def default_analysis_kwargs():
    """Default kwargs for analyze_results matching typical pipeline usage."""
    return {
        "config_columns": [
            "task_definition.seq_len",
            "optimization.lr",
            "run_name",
            MODEL_SEED_COL,
        ],
        "dropped_columns": [],
        "reporting_metrics": ["mse", "loss", "f1", "accuracy"],
        "metric_prefixes": ["val/", "test/", "val_best_rerun/"],
        "optimization_col": None,
        "optimization_mode": "min",
        "required_data_seeds": None,
        "required_model_seeds": None,
        "additional_ignored_cols": None,
    }
