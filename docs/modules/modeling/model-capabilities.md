# Model Capabilities

This page is the contributor-facing inventory of public model configs under
`configs/model/`. The source of truth is `picid.model.catalog.MODEL_CATALOG`.

Canonical development entry points are:

- `picid.model.forecasters` for forecasting model families
- `picid.model.estimators` for non-forecaster model families
- `picid.model.adapters.base` for shared adapter base classes

`window_average` is the preferred family name for the `naive`/`mean` aliases,
and `persistence`/`similar_period` are stale legacy configs kept outside the
canonical inventory until real implementations exist.

| Config | Canonical Class | Family | Execution Style | Regression | Classification | Forecasting | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `carte_fit_predict` | `picid.model.estimators.carte.wrapper.FitPredictCarteWrapper` | `carte` | `fit_predict_wrapper` | yes | yes | no | CARTE-based tabular wrapper with text-aware preprocessing. |
| `catboost` | `catboost.CatBoostRegressor` | `catboost` | `sklearn_estimator` | yes | no | no | Direct CatBoost regressor target; forecasting is not part of this config. |
| `cnn_1d` | `picid.model.estimators.cnn1d.wrapper.CNN1D_Wrapper` | `cnn1d` | `feed_forward_wrapper` | yes | yes | no | 1D CNN encoder wrapped for supervised PHM tasks. |
| `crossformer` | `picid.model.forecasters.crossformer_model.Crossformer_Forecaster` | `crossformer` | `lightning_forecaster` | yes | yes | yes | Transformer forecaster with decoder-style task support. |
| `drift` | `picid.model.estimators.drift.wrapper.DriftModelWrapper` | `drift` | `forecasting_wrapper` | no | no | yes | Simple trend-following baseline for forecasting only. |
| `exponential_regression` | `picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper` | `linear_regression` | `forecasting_wrapper` | no | no | yes | Statistical baseline alias with exponential growth mode. |
| `isolation_forest_fit_predict` | `picid.model.estimators.isolation_forest.wrapper.FitPredictIsolationForestWrapper` | `isolation_forest` | `fit_predict_wrapper` | no | yes | no | Anomaly detection mapped into the classification family. |
| `linear_forecaster` | `picid.model.forecasters.linear_model.linear_model.Linear_Forecaster` | `linear_forecaster` | `lightning_forecaster` | no | no | yes | Minimal linear forecaster for time-series prediction. |
| `linear_regression` | `picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper` | `linear_regression` | `feed_forward_wrapper` | yes | yes | no | Flattened statistical baseline for supervised tasks. |
| `lstm` | `picid.model.forecasters.lstm_model.LSTM_Forecaster` | `lstm` | `lightning_forecaster` | yes | yes | yes | LSTM forecaster/prognostics model; also used for state forecasting. |
| `mean` | `picid.model.estimators.window_average.wrapper.WindowAverageWrapper` | `window_average` | `forecasting_wrapper` | no | no | yes | Window-average alias; canonical family is window_average. |
| `mlp` | `picid.model.estimators.mlp.wrapper.MLPWrapper` | `mlp` | `feed_forward_wrapper` | yes | yes | no | Fully connected baseline for supervised PHM tasks. |
| `naive` | `picid.model.estimators.window_average.wrapper.WindowAverageWrapper` | `window_average` | `forecasting_wrapper` | no | no | yes | Last-value alias; canonical family is window_average. |
| `patchtst` | `picid.model.forecasters.patchtst_model.PatchTST_Forecaster` | `patchtst` | `lightning_forecaster` | yes | yes | yes | PatchTST forecaster with regression/classification support. |
| `polynomial_regression` | `picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper` | `linear_regression` | `forecasting_wrapper` | no | no | yes | Statistical baseline alias with polynomial growth mode. |
| `ses` | `picid.model.estimators.ses.wrapper.SESModelWrapper` | `ses` | `forecasting_wrapper` | no | no | yes | Simple exponential smoothing baseline for forecasting only. |
| `stf` | `picid.model.forecasters.spacetimeformer_model.Spacetimeformer_Forecaster` | `spacetimeformer` | `lightning_forecaster` | yes | yes | yes | Spacetimeformer supports forecasting plus decoder-style tasks. |
| `tabdpt_fit_predict` | `picid.model.estimators.tabdpt.wrapper.FitPredictTabDPTWrapper` | `tabdpt` | `fit_predict_wrapper` | yes | yes | no | TabDPT tabular wrapper for regression and classification. |
| `tabpfn_fit_predict` | `picid.model.estimators.tabpfn.wrapper.FitPredictTabPFNWrapper` | `tabpfn` | `fit_predict_wrapper` | yes | yes | no | TabPFN tabular wrapper for regression and classification. |
| `tide` | `picid.model.forecasters.tide_model.TiDE_Forecaster` | `tide` | `lightning_forecaster` | yes | yes | yes | TiDE forecaster; also supports state forecasting in code paths. |
| `timeseries_transformer` | `picid.model.forecasters.timeseries_transformer_model.Timeseries_Transformer_Forecaster` | `timeseries_transformer` | `lightning_forecaster` | yes | yes | yes | Vanilla Transformer forecaster with regression/classification support. |
| `xgboost_fit_predict` | `picid.model.estimators.xgboost.wrapper.FitPredictXGBoostWrapper` | `xgboost` | `fit_predict_wrapper` | yes | yes | no | Gradient-boosted sklearn baseline for tabular tasks. |
