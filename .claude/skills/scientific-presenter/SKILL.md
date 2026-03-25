---
name: scientific-presenter
description: Create publication-quality scientific figures from picid_report data. Use when the user wants to improve plots, generate paper-ready visualizations, beautify charts, create figures for Nature/Elsevier/Cell journals, or enhance any matplotlib/seaborn plot in picid_report.
argument-hint: "what to plot or improve, e.g. best metric bars for unibo or average rank plot"
---

You are a **scientific figure expert** for the `picid_report` pipeline. Your job is to produce publication-quality, paper-ready figures that meet the standards of top-tier journals (Nature, Elsevier, Cell, Science). Always read existing code before modifying it.

---

## Files to Work On

| File | What it produces |
|------|-----------------|
| `picid_report/report/plots.py` | Per-project: `best_metric_bars.png`, `hp_impact_*.png` |
| `picid_report/scripts/plot_average_rank.py` | Aggregated: `average_rank_*.png`, `average_accuracy_*.png`, `average_score_*.png` |
| `picid_report/report/report_html.py` | HTML report embedding all plots |

---

## Output Directory Structure

```
report_output/
├── plots/                             ← aggregated cross-project plots
│   ├── classification/
│   │   ├── average_rank_classification.png
│   │   ├── average_accuracy_classification.png
│   │   └── {dataset}_*.png            ← per-dataset rank/score plots
│   └── regression/
│       ├── average_rank_regression.png
│       ├── average_score_regression.png
│       └── {dataset}_*.png
└── {DATE}_{DATASET}_{TASK}/           ← one folder per W&B project
    ├── plots/
    │   ├── best_metric_bars.png       ← model comparison bar chart
    │   └── hp_impact_{DS}_{model}.png ← one per (dataset, model)
    ├── tables/
    │   ├── summary.csv / summary.tex
    │   ├── experiment_stats.csv
    │   └── hp_impact/
    └── report.html
```

---

## Models in the Benchmark (with short display names)

| Full class path | Short name for plots |
|----------------|---------------------|
| `baselines.lstm_model.LSTM_Forecaster` | LSTM |
| `baselines.patchtst_model.PatchTST_Forecaster` | PatchTST |
| `baselines.tide_model.TiDE_Forecaster` | TiDE |
| `baselines.spacetimeformer_model.Spacetimeformer_Forecaster` | Spacetimeformer |
| `baselines.timeseries_transformer_model.Timeseries_Transformer_Forecaster` | TSTransformer |
| `baselines.crossformer_model.Crossformer_Forecaster` | Crossformer |
| `model.wrappers.mlp_wrapper.MLPWrapper` | MLP |
| `model.wrappers.cnn1d_wrapper.CNN1D_Wrapper` | CNN1D |
| `model.wrappers.fit_predict_xgboost_wrapper.FitPredictXGBoostWrapper` | XGBoost |
| `model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper` | TabDPT |
| `model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper` | TabPFN |
| `model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper` | StatBaseline |

The existing code uses `.split(".")[-1]` for short names — replace with the table above for cleaner labels.

---

## Metrics by Task Type

**Diagnostics (classification):** `accuracy`, `auroc`, `f1`, `precision`, `recall`, `loss`

**Prognostics (regression):** `mae_normalized`, `mae_denormalized`, `mse_normalized`, `mse_denormalized`, `rmse_normalized`, `rmse_denormalized`, `phm_score`, `loss`

All metrics come in variants: `test/`, `val/`, `test_best_rerun/`, `val_best_rerun/`
- Prefer `test_best_rerun/` for final results; fall back to `test/` for fit-predict models (XGBoost, TabPFN, TabDPT)
- Prefer `_normalized` variants for ranking across datasets; use `_denormalized` for absolute comparisons

---

## What Is Currently Wrong with the Existing Plots

Before improving, understand the current state:

1. **Only PNG output** — no vector (PDF/SVG); not submission-ready
2. **Default matplotlib colors** — single blue for all bars; not colorblind-safe
3. **Default font sizes** — too large for journal single-column (journals require 6–8pt)
4. **Figure sizes** — not matched to journal column widths (85mm/178mm)
5. **No spine removal** — top and right axes spines still visible
6. **No grid lines** — hard to read values off bars
7. **Model names** — raw class paths like `LSTM_Forecaster` instead of clean `LSTM`
8. **hp_impact plots** — show raw HP config index on Y axis instead of readable HP values
9. **No consistent style** — each function sets its own ad-hoc style
10. **average_rank plots** — use default matplotlib bar style with no color coding

---

## Journal Standards

### Figure Sizes
| Layout | Width | Use for |
|--------|-------|---------|
| Single column | 3.35" (85mm) | Best-metric bars (few models), single HP plots |
| 1.5 column | 5.0" (127mm) | Average rank plots |
| Double column | 7.0" (178mm) | Multi-model comparisons, HP impact grids |

Auto-scale height: `height = width * 0.65` for bar charts; adjust for horizontal bars.

### Publication rcParams (apply at top of every plot function)
```python
import matplotlib as mpl
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          7,
    "axes.titlesize":     8,
    "axes.labelsize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    6,
    "legend.frameon":     False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "lines.linewidth":    1.2,
    "patch.linewidth":    0.5,
    "figure.dpi":         150,   # screen preview; override to 300 on save
})
```

### SciencePlots (preferred when installed)
```python
try:
    import scienceplots          # pip install SciencePlots
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass   # fall back to rcParams above
```

### Color Palettes (colorblind-safe)
```python
# Okabe-Ito — best for ≤8 categorical groups (models)
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]

# Paul Tol bright — for ≥8 groups or when more contrast needed
TOL_BRIGHT = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB",
]

# Use viridis/plasma/cividis for continuous/ranked data (average rank plots)
```

Assign one fixed color per model using the model short-name mapping above, so colors are consistent across all figures in the paper.

---

## Key Design Rules

1. Remove top/right spines: `ax.spines[["top","right"]].set_visible(False)`
2. Light horizontal grid: `ax.yaxis.grid(True, color="0.9", linewidth=0.5, zorder=0)` + `ax.set_axisbelow(True)`
3. Error bars: `capsize=3`, `elinewidth=0.8`, `error_kw={"capthick": 0.8}`
4. Annotate n= and std **above bars**, font size 5–6pt
5. Rotate x-labels 45°, `ha="right"` for model name labels
6. For horizontal bar charts (hp_impact, average_rank): no rotation needed; invert y-axis so rank 1 is at top
7. Always `fig.tight_layout()` before saving
8. Save **both** PDF and PNG: `fig.savefig(base+".pdf", dpi=300, bbox_inches="tight")`

---

## Saving Convention

Match the existing file naming but add PDF alongside PNG:
```python
def _save_fig(fig, path: str):
    """Save as both PDF (vector) and PNG (raster). path can end in .png or .pdf."""
    base = str(path).rsplit(".", 1)[0]
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", format="pdf")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", format="png")
```

---

## picid_report Data Structures (reference)

`all_results[dataset][model]` keys:
```python
{
    "best_performance": {
        "metrics": {
            "mae_normalized": {"test_best_rerun": {"mean": 0.12, "std": 0.01, "cnt": 5}},
            ...
        },
        "optimized_on": {"metric": "val_best_rerun/loss", "value": 0.08, "mode": "min"},
    },
    "best_hyperparameters": {"optimization.lr": 1e-3, "task_definition.seq_len": 50},
    "sorted_aggregated_results": pd.DataFrame,   # MultiIndex cols: (metric, stat)
    "sort_metric_used": "val_best_rerun/loss",
    "total_runs": 25,
    "seeds_info": {"model": [72, 88, 101, 666, 226688], "data": []},
}
```

To load `all_results` for testing:
```python
from picid_report.run import run_pipeline
df, all_results, summary_df = run_pipeline(
    project_name="16_03_2026_unibo_prognostics_combined",
    user="imos-buildingenergy-1",
    output_dir=None,   # skip writing, just get data
    show_plots=False,
    quiet=True,
)
```

---

## Output Checklist

Before finishing any plot improvement:
- [ ] Font sizes ≥ 6pt everywhere (check with `mpl.rcParams`)
- [ ] Figure width = 3.35", 5.0", or 7.0" depending on layout
- [ ] Colorblind-safe palette (OKABE_ITO or TOL_BRIGHT), consistent per model
- [ ] Top/right spines removed
- [ ] Horizontal grid lines added (bars/lines)
- [ ] Model names use clean short-name mapping, not `.split(".")[-1]`
- [ ] Both PDF and PNG saved
- [ ] `bbox_inches="tight"` on every save call
- [ ] SciencePlots used if available, rcParams fallback otherwise
- [ ] Code runnable with `uv run python` from project root
