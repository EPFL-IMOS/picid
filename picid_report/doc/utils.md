# utils.py — Shared formatting and column flattening

**Navigation:** [← Documentation index](README.md)

This document describes **format_mean_std_count** and **flatten_aggregated_columns** with examples.

---

## 1. format_mean_std_count(mean, std, count, precision=4)

Formats aggregate stats as "mean ± std (n=count)". NaN std is shown as 0. count can be int or str (e.g. "n/a").

**Example:**
```python
from picid_report.utils import format_mean_std_count

format_mean_std_count(1.2345, 0.0012, 3, precision=4)
# "1.2345 ± 0.0012 (n=3)"
format_mean_std_count(0.5, float("nan"), 5, precision=2)
# "0.50 ± 0.00 (n=5)"
format_mean_std_count(0.1, 0.0, "n/a", precision=2)
# "0.10 ± 0.00 (n=n/a)"
```

---

## 2. flatten_aggregated_columns(df)

Converts MultiIndex columns like (metric, "mean"), (metric, "std"), (metric, "count") into metric_mean, metric_std, metric_count. Plain columns unchanged. Returns a copy.

**Example:**
```python
from picid_report.utils import flatten_aggregated_columns

# After groupby.agg you get tuple columns; flatten for sorting/display
flat = flatten_aggregated_columns(aggregated_df)
# flat has columns like "val_best_rerun/loss_mean", "val_best_rerun/loss_std", ...
```

---

## 3. Where used

- format_mean_std_count: reporting.py (summary table, HP impact cells).
- flatten_aggregated_columns: analysis.py (after aggregate), reporting.py (_build_hp_impact_df).

---

**Navigation:** [← Documentation index](README.md)
