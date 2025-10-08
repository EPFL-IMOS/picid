# How to add a new Model

# Fit-predict Style
On the example of `TabDPT` and the dataset `railway`.
Create the necessary config files.

```
experiment/railway_traction/forecasting/tabdpt_fit_predict.yaml
model/tabdpt.yaml
model_config/forecasting/tabdpt_fit_predict
```

Add the model to `lmetk.model.wrappers`.
