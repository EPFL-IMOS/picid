"""
Search space configuration: single place for per-dataset, default-for-all, or data-first.

Resolution order (in get_search_space(dataset, model)):
  1. Per-dataset: EXPECTED_SEARCH_SPACE[dataset][model] if present.
  2. Default for all datasets: DEFAULT_SEARCH_SPACE[model] if DEFAULT_SEARCH_SPACE is set.
  3. Otherwise None → pipeline uses data-first (infer varying HPs from data).

So you can:
  - Define per (dataset, model) in EXPECTED_SEARCH_SPACE → schema-first for those.
  - Define model-only grid in DEFAULT_SEARCH_SPACE → same grid for all datasets when no per-dataset entry.
  - Leave both empty / no match → data-first (infer from data).

Backward compatibility: if DEFAULT_SEARCH_SPACE is None (not set), get_search_space still
checks config.EXPECTED_SEARCH_SPACE so existing config.py content continues to act as the default.
Set DEFAULT_SEARCH_SPACE = {} explicitly to disable that and get data-first when no per-dataset entry.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Dataset-first organization: {dataset: {model: {hp: [values]}}}
# Start empty - will be populated as needed
EXPECTED_SEARCH_SPACE: Dict[str, Dict[str, Dict[str, List[Any]]]] = {
    # Example structure:
    # "nb14": {
    #     "baselines.lstm_model.LSTM_Forecaster": {
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     },
    # },
    # "mzvav": {
    #     "baselines.lstm_model.LSTM_Forecaster": {
    #         "task_definition.seq_len": [1, 5, 10, 20],  # Different for diagnostics
    #         "optimization.lr": [0.001, 0.0005],
    #     },
    # },
    # "UNIBO21":
    # {
    #     "baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": {
    #         "task_definition.seq_len": [1, 10, 50, 100, 1000],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     },
    #     "baselines.patchtst_model.PatchTST_Forecaster": {
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #     },
    #     "baselines.lstm_model.LSTM_Forecaster": {
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     },
    #     "model.wrappers.mlp_wrapper.MLPWrapper": {
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     },
    #     "model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": {
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     },
    #     "baselines.tide_model.TiDE_Forecaster": {
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     },
    #     "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": {
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     },
    #     "baselines.crossformer_model.Crossformer_Forecaster": {
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     },
    #     "baselines.spacetimeformer_model.Spacetimeformer_Forecaster": {
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     },
    #     "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": {
    #         "task_definition.seq_len": [1, 10, 50, 100],
    #         "optimization.lr": [0.001, 0.0005, 0.0001],
    #     }
    #     },
    # "PHME20":
    # {
    # "baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": {
    #     "task_definition.subset_ratio": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     "dataset._overrides.train.dataset_cfg.subset_seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     "dataset._overrides.train.dataset_cfg.subset_ratio": [0.001, 0.0005, 0.0001],
    # },
    # "baselines.patchtst_model.PatchTST_Forecaster": {
    #     "optimization.lr": [0.001, 0.0005, 0.0001],
    #     "task_definition.seq_len": [1, 10, 50, 100],
    # },
    # "baselines.lstm_model.LSTM_Forecaster": {
    #     "task_definition.seq_len": [1, 10, 50, 100],
    #     "optimization.lr": [0.001, 0.0005, 0.0001],
    # },
    # "model.wrappers.mlp_wrapper.MLPWrapper": {
    #     "task_definition.seq_len": [1, 10, 50, 100],
    #     "optimization.lr": [0.001, 0.0005, 0.0001],
    # },
    # "model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": {
    #     "task_definition.seq_len": [1, 10, 50, 100],
    #     "optimization.lr": [0.001, 0.0005, 0.0001],
    # },
    # "baselines.tide_model.TiDE_Forecaster": {
    #     "task_definition.seq_len": [1, 10, 50, 100],
    #     "optimization.lr": [0.001, 0.0005, 0.0001],
    # },
    # "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": {
    #     "task_definition.seq_len": [1, 10, 50, 100],
    #     "optimization.lr": [0.001, 0.0005, 0.0001],
    # },
    # "baselines.crossformer_model.Crossformer_Forecaster": {
    #     "task_definition.seq_len": [1, 10, 50, 100],
    #     "optimization.lr": [0.001, 0.0005, 0.0001],
    # },
    # "baselines.spacetimeformer_model.Spacetimeformer_Forecaster": {
    #     "task_definition.seq_len": [1, 10, 50, 100],
    #     "optimization.lr": [0.001, 0.0005, 0.0001],
    # },
    # "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": {
    #     "task_definition.seq_len": [1, 10, 50, 100],
    #     "optimization.lr": [0.001, 0.0005, 0.0001],
    # }
    # },
}

# Default (model-only) grid used for all datasets when no per-dataset entry exists.
# Shape: {model: {hp_name: [values]}}. Set to {} to force data-first when no per-dataset match.
# Leave None for backward compatibility (then config.EXPECTED_SEARCH_SPACE is used if present).
DEFAULT_SEARCH_SPACE: Optional[Dict[str, Dict[str, List[Any]]]] = None

# EXPECTED_SEARCH_SPACE =  {
#     "baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": {
#         "task_definition.seq_len": [1, 10, 50, 100],
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#     },
#     "baselines.patchtst_model.PatchTST_Forecaster": {
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#         "task_definition.seq_len": [1, 10, 50, 100],
#     },
#     "baselines.lstm_model.LSTM_Forecaster": {
#         "task_definition.seq_len": [1, 10, 50, 100],
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#     },
#     "model.wrappers.mlp_wrapper.MLPWrapper": {
#         "task_definition.seq_len": [1, 10, 50, 100],
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#     },
#     "model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": {
#         "task_definition.seq_len": [1, 10, 50, 100],
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#     },
#     "baselines.tide_model.TiDE_Forecaster": {
#         "task_definition.seq_len": [1, 10, 50, 100],
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#     },
#     "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": {
#         "task_definition.seq_len": [1, 10, 50, 100],
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#     },
#     "baselines.crossformer_model.Crossformer_Forecaster": {
#         "task_definition.seq_len": [1, 10, 50, 100],
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#     },
#     "baselines.spacetimeformer_model.Spacetimeformer_Forecaster": {
#         "task_definition.seq_len": [1, 10, 50, 100],
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#     },
#     "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": {
#         "task_definition.seq_len": [1, 10, 50, 100],
#         "optimization.lr": [0.001, 0.0005, 0.0001],
#     },
#     # "model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper": {
#     #     "task_definition.seq_len": [1, 5, 10, 20, 50, 100],
#     #     "task_definition.stride_train": [1, 5, 50, 100],
#     # },
#     # "model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper": {
#     #     "task_definition.seq_len": [1, 5, 10, 20, 50, 100],
#     #     "task_definition.stride_train": [1, 5, 50, 100],
#     # },

#     # "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": {
#     #     "task_definition.seq_len": [1, 10, 50, 100, 150],
#     #     'optimization.lr': [0.001, 0.0005, 0.0001],,
#     # },
#     # FitPredictTabPFNWrapper is omitted, so it will use automatic discovery
# }


def _legacy_default_search_space() -> Optional[Dict[str, Dict[str, List[Any]]]]:
    """Backward compat: config.EXPECTED_SEARCH_SPACE when DEFAULT_SEARCH_SPACE is not set."""
    try:
        from picid_report import config

        return (
            config.EXPECTED_SEARCH_SPACE
            if getattr(config, "EXPECTED_SEARCH_SPACE", None)
            else None
        )
    except Exception:
        return None


def get_search_space(
    dataset: str,
    model: str,
    use_default: bool = True,
) -> Optional[Dict[str, List[Any]]]:
    """
    Resolve search space for (dataset, model). Single entry point: per-dataset, then default, then None.

    1. Per-dataset: EXPECTED_SEARCH_SPACE[dataset][model]
    2. If use_default: default for all datasets (DEFAULT_SEARCH_SPACE[model] or config.EXPECTED_SEARCH_SPACE[model])
    3. None → data-first (infer varying HPs from data).

    Parameters
    ----------
    dataset : str
        Dataset name
    model : str
        Model name/identifier
    use_default : bool, default True
        If False, skip step 2 (only per-dataset or None). Use for --data-first when no per-dataset entry.

    Returns
    -------
    Optional[Dict[str, List[Any]]]
        HP name -> list of values, or None for data-first
    """
    # 1) Per-dataset
    if dataset in EXPECTED_SEARCH_SPACE and model in EXPECTED_SEARCH_SPACE[dataset]:
        out = EXPECTED_SEARCH_SPACE[dataset][model]
        logger.debug(
            "get_search_space(%s, %s) -> per-dataset, %s keys",
            dataset,
            model,
            list(out.keys()) if out else None,
        )
        return out
    if not use_default:
        logger.debug(
            "get_search_space(%s, %s) -> None (data-first, use_default=False)",
            dataset,
            model,
        )
        return None
    # 2) Default (model-only): explicit DEFAULT_SEARCH_SPACE or legacy config
    default = (
        DEFAULT_SEARCH_SPACE
        if DEFAULT_SEARCH_SPACE is not None
        else _legacy_default_search_space()
    )
    if default and model in default:
        out = default[model]
        logger.debug(
            "get_search_space(%s, %s) -> default (all datasets), %s keys",
            dataset,
            model,
            list(out.keys()) if out else None,
        )
        return out
    logger.debug("get_search_space(%s, %s) -> None (data-first)", dataset, model)
    return None


def get_model_grid_from_search_space(
    dataset: str,
    model: str,
    search_space: Optional[Dict[str, Any]],
) -> Optional[Dict[str, List[Any]]]:
    """
    Get the HP grid for (dataset, model) from a search space dict.

    Accepts either structure so callers do not need to detect format:
    - New: {dataset: {model: {hp: [values]}}}
    - Legacy: {model: {hp: [values]}}

    Returns None if not found or search_space is None/empty (triggers auto-discovery).

    Parameters
    ----------
    dataset : str
        Dataset name
    model : str
        Model name/identifier
    search_space : dict or None
        Either structure; can be from config.EXPECTED_SEARCH_SPACE or PipelineConfig.

    Returns
    -------
    Optional[Dict[str, List[Any]]]
        HP name -> list of values, or None for auto-discovery
    """
    if not search_space or not isinstance(search_space, dict):
        return None
    sample_val = next(iter(search_space.values()), None)
    if not sample_val or not isinstance(sample_val, dict):
        return None
    sample_inner = next(iter(sample_val.values()), None)
    is_legacy = isinstance(sample_inner, list)
    if is_legacy:
        return search_space.get(model)
    if dataset in search_space and model in search_space[dataset]:
        return search_space[dataset][model]
    return None
