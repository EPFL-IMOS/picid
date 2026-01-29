# lmetk_report/config.py
"""
Central configuration file for the experiment analysis project.

This file contains constants and configurations that are used across the
different modules, making it easy to update key parameters in one place.
"""

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
