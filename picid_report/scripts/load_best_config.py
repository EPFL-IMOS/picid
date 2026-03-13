"""
Load best hyperparameter config from a report's hp_impact CSV files.

The report exports one CSV per (dataset, model) under tables/hp_impact/.
The first row of each CSV is the best config (sorted by sort metric); HP columns
e.g. task_definition.seq_len, optimization.lr are read from that row.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _parse_hp_impact_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse hp_impact_{dataset}_{model}.csv into (dataset, model).
    Dataset is taken as the first segment before underscore; rest is model (model names can contain dots/underscores).
    """
    if not filename.startswith("hp_impact_") or not filename.endswith(".csv"):
        return None
    inner = filename[len("hp_impact_") : -len(".csv")]
    parts = inner.split("_", 1)
    if len(parts) != 2:
        return None
    return (parts[0], parts[1])


def list_report_models(report_dir: str | Path) -> List[Tuple[str, str]]:
    """
    List (dataset, model) pairs from report_output/.../tables/hp_impact/*.csv.

    Returns
    -------
    List of (dataset, model) where model is the sanitized name as in the filename.
    """
    report_dir = Path(report_dir)
    hp_dir = report_dir / "tables" / "hp_impact"
    if not hp_dir.is_dir():
        return []
    out: List[Tuple[str, str]] = []
    for f in hp_dir.glob("hp_impact_*.csv"):
        parsed = _parse_hp_impact_filename(f.name)
        if parsed:
            out.append(parsed)
    return out


def get_best_row_from_hp_impact_csv(csv_path: str | Path) -> Optional[Dict[str, Any]]:
    """
    Read the first (best) row from an hp_impact CSV and return HP values as a flat dict.

    HP columns (e.g. task_definition.seq_len, optimization.lr, task_definition.stride_train)
    are returned as scalar values; metric columns are ignored for reproduction.

    Returns
    -------
    Dict mapping column name -> value for HP columns only, or None if CSV is empty/invalid.
    """
    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, nrows=1)
    except Exception:
        return None
    if df.empty:
        return None

    # HP columns: typically task_definition.seq_len, optimization.lr, (task_definition.stride_train for fit_predict)
    hp_prefixes = ("task_definition.", "optimization.", "datamodule.")
    row = df.iloc[0]
    best: Dict[str, Any] = {}
    for col in df.columns:
        if any(col.startswith(p) for p in hp_prefixes) or col == "seed":
            val = row.get(col)
            if pd.isna(val):
                continue
            try:
                num = float(val)
                best[col] = int(num) if num == int(num) else num
            except (TypeError, ValueError):
                best[col] = val
    return best if best else None


def load_best_configs_from_report(
    report_dir: str | Path,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load best HP config for each (dataset, model) from report's hp_impact CSVs.

    Returns
    -------
    Dict[(dataset, model)] -> dict of HP column -> value (e.g. task_definition.seq_len -> 10).
    """
    report_dir = Path(report_dir)
    hp_dir = report_dir / "tables" / "hp_impact"
    if not hp_dir.is_dir():
        return {}

    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for f in hp_dir.glob("hp_impact_*.csv"):
        parsed = _parse_hp_impact_filename(f.name)
        if not parsed:
            continue
        dataset, model = parsed
        best = get_best_row_from_hp_impact_csv(f)
        if best is not None:
            result[(dataset, model)] = best
    return result
