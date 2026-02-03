# lmetk_report/config.py
"""
Central configuration file for the experiment analysis project.

This file contains constants and configurations that are used across the
different modules, making it easy to update key parameters in one place.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

# Key column names from the wandb logs used to identify models, datasets, etc.
COLUMN_CONFIG = {
    # Column for the model's class path or identifier
    "model_target": "model._target_",
    # Column for the dataset's name
    "dataset_name": "datasource.data_name",
    # Column for the metric being monitored for early stopping (e.g., 'val/loss')
    "optimization_metric": "callbacks.early_stopping.monitor",
    # Column for the optimization mode ('min' or 'max')
    "optimization_mode": "callbacks.early_stopping.mode",
    # --- Explicit Task Definition Columns ---
    "target_metric": "task_definition.target_metric",
    "target_metric_mode": "task_definition.target_metric_mode",
    "evaluator_metrics": "evaluator.train.metric_names",
}

# Names of nested dictionary columns from wandb to be flattened into separate columns.
COLUMNS_TO_NORMALIZE = [
    "optimizer",
    "optimization",
    "datasource",
    "task_definition",
    "model",
    "dataset",
    "datamodule",
    "callbacks",
    "paths",
    "evaluator",
    "cache",
    "logger",
    "trainer",
    "_wandb",
]

# Columns to exclude from being considered as varying hyperparameters, even if they vary.
# This typically includes run-specific seeds or identifiers.
SPECIAL_COLUMNS = [
    "datasource.parameters.data_seed",
    "task_definition.subset_seed",
    "datamodule.subset_seed",
    "seed",
]

# Column name filters for columns to be dropped entirely during data loading.
# This is useful for removing noisy or irrelevant configuration groups.
COLUMN_FILTERS_TO_DROP = ["paths."]

# Filters for columns to IGNORE when searching for varying hyperparameters.
# These columns are kept in the DataFrame but are not treated as HPs (e.g., results, timers).
COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH = [
    "run_name",
    "_runtime",
    "_step",
    "_timestamp",
    "epoch",
    "val/",  # Ignore all validation metrics
    "val_best_rerun/",
    "test/",  # Ignore all test metrics
    "train/",  # Ignore all training metrics
    "AvgTime/",  # Ignore all timing metrics
    "plots/",  # Ignore all plot outputs
    "callbacks.model_checkpoint.",
    "model.model_cache_path",
    # Repeated config parameters
    "model.max_seq_len",
    "dataset.dataset_cfg.seq_len",
    "optimization.optimizer.lr",
    "optimization.scheduler.peak_lr",
    # Logger-specific columns
    "logger.wandb.id",
    "logger.wandb.save_dir",
    # Trainer-specific columns
    "trainer.default_root_dir",
    "_wandb.runtime",
    "cache.use_preprocessing_file_lock",
]

ADDITIONAL_COLUMNS_TO_SHOW = [
    "epoch",
    "AvgTime/train_epoch_mean",
    "AvgTime/val_epoch_mean",
]

# The column name in the DataFrame that holds the path to the best checkpoint for a run.
# The script will try this name first before searching for common alternatives.
# UPDATE THIS if you know the exact column name in your logs.
CHECKPOINT_PATH_COLUMN = "callbacks.model_checkpoint.dirpath"

# Default column for Data Seeds (e.g., shuffling, splits)
DEFAULT_DATA_SEED_COL = "datasource.parameters.data_seed"

# Default column for Model/Training Seeds (e.g., initialization)
# UPDATE THIS to match your specific log column (often just "seed" or "model.seed")
DEFAULT_MODEL_SEED_COL = "seed"


# --- Pipeline Requirements ---
# These columns MUST exist in the DataFrame after normalization for the
# analysis and reporting to work correctly.
REQUIRED_COLUMNS = [
    COLUMN_CONFIG["model_target"],
    COLUMN_CONFIG["dataset_name"],
    # Seeds
    # DEFAULT_DATA_SEED_COL, # For now we vary over the model instantiation seed only
    DEFAULT_MODEL_SEED_COL,
    # Checkpointing
    CHECKPOINT_PATH_COLUMN,
]


#  Global Expected Search Space ---
# If None, the script only shows completed experiments.
# If provided, it creates a grid and shows gaps as NaNs.
EXPECTED_SEARCH_SPACE = {
    "baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": {
        "task_definition.seq_len": [1, 10, 50, 100, 150],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
    "baselines.patchtst_model.PatchTST_Forecaster": {
        "optimization.lr": [0.001, 0.0005, 0.0001],
        "task_definition.seq_len": [1, 10, 50, 100, 150],
    },
    "baselines.lstm_model.LSTM_Forecaster": {
        "task_definition.seq_len": [1, 10, 50, 100, 150],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
    "model.wrappers.mlp_wrapper.MLPWrapper": {
        "task_definition.seq_len": [1, 10, 50, 100, 150],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
    "model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": {
        "task_definition.seq_len": [1, 10, 50, 100, 150],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
    "baselines.tide_model.TiDE_Forecaster": {
        "task_definition.seq_len": [1, 10, 50, 100, 150],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
    "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": {
        "task_definition.seq_len": [1, 10, 50, 100, 150],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
    "baselines.crossformer_model.Crossformer_Forecaster": {
        "task_definition.seq_len": [1, 10, 50, 100, 150],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
    "baselines.spacetimeformer_model.Spacetimeformer_Forecaster": {
        "task_definition.seq_len": [1, 10, 50, 100, 150],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
    "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": {
        "task_definition.seq_len": [1, 10, 50, 100, 150],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
    # "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": {
    #     "task_definition.seq_len": [1, 10, 50, 100, 150],
    #     'optimization.lr': [0.001, 0.0005, 0.0001],,
    # },
    # FitPredictTabPFNWrapper is omitted, so it will use automatic discovery
}


@dataclass(frozen=False)
class PipelineConfig:
    """
    Bundled configuration for the analysis pipeline. Pass to analyze_results
    (and optionally load_runs_df) to override module-level config without
    mutating global state.
    """

    column_config: Dict[str, str]
    expected_search_space: Dict[str, Dict[str, List[Any]]]
    column_filters_to_ignore_for_hp_search: List[str]
    special_columns: List[str]
    columns_to_normalize: List[str]
    column_filters_to_drop: List[str]

    @classmethod
    def from_default(cls) -> "PipelineConfig":
        """Build a PipelineConfig from the current module-level config."""
        return cls(
            column_config=dict(COLUMN_CONFIG),
            expected_search_space={
                k: {kk: list(vv) for kk, vv in v.items()}
                for k, v in EXPECTED_SEARCH_SPACE.items()
            },
            column_filters_to_ignore_for_hp_search=list(
                COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH
            ),
            special_columns=list(SPECIAL_COLUMNS),
            columns_to_normalize=list(COLUMNS_TO_NORMALIZE),
            column_filters_to_drop=list(COLUMN_FILTERS_TO_DROP),
        )
