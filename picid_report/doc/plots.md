# plots.py — Bar charts and HP impact plots

**Navigation:** [← Documentation index](README.md)

**Module:** `picid_report.report.plots`

This document describes the **plotting** helpers that read from **all_results** (output of **analyze_results**): **plot_best_metric_bars** and **plot_hp_impact**. Both require **matplotlib**; if it is not installed, they raise a clear ImportError.

---

## 1. plot_best_metric_bars(all_results, metric="test/mse", save_path=None, figsize=None)

Builds a **bar chart** of the best value of the given metric for each (dataset, model) combination. Each bar is one model/dataset; the height is the mean from **best_performance.metrics**. Skips combinations that do not have the requested metric/prefix.

**Signature:**

```python
def plot_best_metric_bars(
    all_results: defaultdict,
    metric: str = "test/mse",
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> "object":
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| **all_results** | Result of **analyze_results** (dataset -> model -> result dict). |
| **metric** | Metric in prefix/name form, e.g. `"test/mse"`, `"val_best_rerun/loss"`. Default `"test/mse"`. |
| **save_path** | If set, save the figure to this path. |
| **figsize** | (width, height) in inches; default auto-sized from number of bars. |

**Returns:** Matplotlib figure, or **None** if no data for the metric.

---

### Example 1: Default metric, display only

```python
from picid_report.plots import plot_best_metric_bars

fig = plot_best_metric_bars(all_results)
if fig is not None:
    import matplotlib.pyplot as plt
    plt.show()
    plt.close(fig)
```

---

### Example 2: Save to file

```python
fig = plot_best_metric_bars(
    all_results,
    metric="val_best_rerun/loss",
    save_path="report_output/plots/best_metric_bars.png",
)
if fig is not None:
    import matplotlib.pyplot as plt
    plt.close(fig)
```

---

### Example 3: Custom figsize

```python
fig = plot_best_metric_bars(all_results, metric="test/mae", figsize=(12, 6))
```

---

### Example 4: No data (returns None)

```python
fig = plot_best_metric_bars(all_results, metric="nonexistent/metric")
# fig is None if no model/dataset has best_performance.metrics["metric"]["prefix"]
```

---

## 2. plot_hp_impact(all_results, model, dataset, metric, save_path=None, figsize=None, max_configs=50)

Builds a **horizontal bar chart** of the given metric vs HP configuration for **one** (dataset, model). Uses **sorted_aggregated_results**; each bar is one HP configuration (ranked). Useful to visualize HP impact for a single model/dataset.

**Signature:**

```python
def plot_hp_impact(
    all_results: defaultdict,
    model: str,
    dataset: str,
    metric: str,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    max_configs: int = 50,
) -> "object":
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| **all_results** | Result of **analyze_results**. |
| **model** | Model key (e.g. `"LSTM"`). |
| **dataset** | Dataset key (e.g. `"UNIBO21"`). |
| **metric** | Metric to plot (e.g. `"test/mse"`); must match a `_mean` column in flattened aggregated results. |
| **save_path** | If set, save figure to this path. |
| **figsize** | (width, height) in inches; default auto-sized. |
| **max_configs** | Maximum number of HP configs to show (default 50) to avoid huge figures. |

**Returns:** Matplotlib figure, or **None** if dataset/model not in all_results, no sorted_aggregated_results, or metric_mean column missing.

---

### Example 1: One model/dataset, display only

```python
from picid_report.plots import plot_hp_impact

fig = plot_hp_impact(all_results, model="LSTM", dataset="UNIBO21", metric="test/mse")
if fig is not None:
    import matplotlib.pyplot as plt
    plt.show()
    plt.close(fig)
```

---

### Example 2: Save and limit configs

```python
fig = plot_hp_impact(
    all_results,
    model="LSTM",
    dataset="UNIBO21",
    metric="val_best_rerun/loss",
    save_path="report_output/plots/hp_impact_UNIBO21_LSTM.png",
    max_configs=30,
)
if fig is not None:
    import matplotlib.pyplot as plt
    plt.close(fig)
```

---

### Example 3: Iterate over all dataset/model (as in run.py)

```python
from picid_report.reporting import iter_hp_impact_tables
from picid_report.plots import plot_hp_impact

for dataset, model, hp_df, metric_used in iter_hp_impact_tables(all_results, precision=4):
    path = f"plots/hp_impact_{dataset}_{model}.png".replace(" ", "_")
    fig = plot_hp_impact(all_results, model=model, dataset=dataset, metric="test/mse", save_path=path)
    if fig is not None:
        import matplotlib.pyplot as plt
        plt.close(fig)
```

---

### Example 4: Missing model/dataset or metric (returns None)

```python
fig = plot_hp_impact(all_results, model="Unknown", dataset="UNIBO21", metric="test/mse")
# fig is None
```

---

## 3. Matplotlib dependency

If matplotlib is not installed, both functions call **\_ensure_matplotlib()** and raise:

```text
ImportError: Plotting requires matplotlib. Install it with: pip install matplotlib
```

Install matplotlib if you use **show_plots=True** or save plots in the pipeline. See [run.md](run.md) for pipeline options.

---

**Navigation:** [← Documentation index](README.md)
