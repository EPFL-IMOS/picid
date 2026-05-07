# PICID

**A unified, config-driven benchmark arena for Prognostics, Health Management (PHM) and time-series forecasting.**

PICID covers the full experiment pipeline in a single run:
data loading → preprocessing → model training → metric evaluation.
Experiments are fully reproducible and composable through Hydra configs.

---

## Install

```bash
git clone <repo-url>
cd PICID
uv sync
```

Set your paths (copy `configs/paths/default.yaml`) and run:

```bash
python picid/run.py paths=<your_config> experiment=unibo/prognostics/raw/cnn_1d
```

→ [Full install and paths setup](getting-started/install.md)

---

## Supported tasks

| Task | Description |
|---|---|
| **Prognostics** | Remaining Useful Life (RUL) estimation from sensor time series |
| **Diagnostics** | Fault classification from sensor windows |
| **Forecasting** | Multi-step ahead time-series prediction |
| **Anomaly detection** | Unsupervised anomaly scoring |

---

## Datasets

| Dataset | Domain | Task |
|---|---|---|
| UNIBO Powertools | Battery degradation | Prognostics |
| C-MAPSS (N-CMAPSS) | Turbofan engine degradation | Prognostics |
| PRONOSTIA | Bearing degradation | Prognostics |
| XJTU-SY | Bearing degradation | Prognostics |
| PHME 2020 | Bearing degradation | Prognostics / Diagnostics |
| HSF-15 | Hydraulic system | Prognostics |
| ThreeW | Oil well operations | Anomaly detection |
| MZVAV | HVAC building system | Forecasting |
| NB14 | Lithium-ion battery | Prognostics |

---

## Models

| Model | Type | Tasks |
|---|---|---|
| CNN-1D | Feed-forward | Prog · Diag |
| LSTM | Feed-forward | Prog · Diag · Forecast |
| MLP | Feed-forward | Prog · Diag |
| PatchTST | Transformer | Prog · Diag · Forecast |
| TiDE | MLP-Mixer | Prog · Diag · Forecast |
| Crossformer | Transformer | Prog · Diag · Forecast |
| Spacetimeformer (STF) | Transformer | Prog · Diag · Forecast |
| TimeLLM | LLM-based | Prog · Diag · Forecast |
| XGBoost | Fit-predict | Prog · Diag · Forecast |
| TabPFN | Foundation model | Prog · Diag · Forecast |
| TabDPT | Foundation model | Prog · Diag · Forecast |
| CARTE | Foundation model | Prog · Diag |
| Linear / Polynomial / Exponential Regression | Baseline | Prog · Forecast |

---

## New here?

1. [Install](getting-started/install.md) — set up the environment and paths
2. [Quickstart](getting-started/quickstart-5min.md) — run your first experiment in 5 minutes
3. [First Run Walkthrough](getting-started/first-run.md) — understand what the run produces
4. [Core Concepts](concepts/index.md) — PHM terminology, system architecture, end-to-end flow

---

## Reference

| Section | What it covers |
|---|---|
| [Interface](interface/index.md) | Components, datasources, transforms, schemas |
| [Modules](modules/index.md) | Data → Transforms → Modeling → Orchestration → Evaluation |
| [API Reference](reference/api/index.md) | Auto-generated signatures and docstrings |
| [How-to Guides](how-to/index.md) | Add datasources, transforms, models, evaluators |
