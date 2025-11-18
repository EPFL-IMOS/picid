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
}

# Names of nested dictionary columns from wandb to be flattened into separate columns.
COLUMNS_TO_NORMALIZE = [
    "optimizer",
    "datasource",
    "task_definition",
    "model",
    "dataset",
    "datamodule",
    "callbacks",
    "paths",
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
]

# The column name in the DataFrame that holds the path to the best checkpoint for a run.
# The script will try this name first before searching for common alternatives.
# UPDATE THIS if you know the exact column name in your logs.
CHECKPOINT_PATH_COLUMN = "callbacks.model_checkpoint.dirpath"
