"""
Expected search space configuration organized by dataset/model.

The pipeline uses this file first: for each (dataset, model) it calls get_search_space(dataset, model).
If that returns a grid, that grid is used (e.g. UNIBO21 gets the grids defined below).
If it returns None, the pipeline falls back to config.EXPECTED_SEARCH_SPACE (legacy model-only)
so other datasets use the same grid as before.

Dict shape: {dataset: {model: {hp_name: [values]}}}.
For datasets/models not listed here, the legacy config (config.py) or auto-discovery is used.
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
    "UNIBO21": 
    {
        "baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster": {
            "task_definition.seq_len": [1, 10, 50, 100, 1000],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        },
        "baselines.patchtst_model.PatchTST_Forecaster": {
            "optimization.lr": [0.001, 0.0005, 0.0001],
            "task_definition.seq_len": [1, 10, 50, 100],
        },
        "baselines.lstm_model.LSTM_Forecaster": {
            "task_definition.seq_len": [1, 10, 50, 100],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        },
        "model.wrappers.mlp_wrapper.MLPWrapper": {
            "task_definition.seq_len": [1, 10, 50, 100],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        },
        "model.wrappers.cnn1d_wrapper.CNN1D_Wrapper": {
            "task_definition.seq_len": [1, 10, 50, 100],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        },
        "baselines.tide_model.TiDE_Forecaster": {
            "task_definition.seq_len": [1, 10, 50, 100],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        },
        "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": {
            "task_definition.seq_len": [1, 10, 50, 100],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        },
        "baselines.crossformer_model.Crossformer_Forecaster": {
            "task_definition.seq_len": [1, 10, 50, 100],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        },
        "baselines.spacetimeformer_model.Spacetimeformer_Forecaster": {
            "task_definition.seq_len": [1, 10, 50, 100],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        },
        "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)": {
            "task_definition.seq_len": [1, 10, 50, 100],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        }
        },
}


def get_search_space(dataset: str, model: str) -> Optional[Dict[str, List[Any]]]:
    """
    Get search space for dataset/model combination from module-level EXPECTED_SEARCH_SPACE.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    model : str
        Model name/identifier
        
    Returns
    -------
    Optional[Dict[str, List[Any]]]
        Search space dict if found, None for auto-discovery
    """
    if dataset in EXPECTED_SEARCH_SPACE:
        if model in EXPECTED_SEARCH_SPACE[dataset]:
            out = EXPECTED_SEARCH_SPACE[dataset][model]
            logger.debug("get_search_space(%s, %s) -> %s keys", dataset, model, list(out.keys()) if out else None)
            return out
    logger.debug("get_search_space(%s, %s) -> None", dataset, model)
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
