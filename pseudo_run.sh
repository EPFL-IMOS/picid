# PHME20 (datasource + transforms from experiment)
uv run python pseudo_run.py experiment=phme20/prognostics/raw/linear_regression

# Optional: disable cache so load + transforms always run
uv run python pseudo_run.py experiment=phme20/prognostics/raw/linear_regression cache.use_cache_after_loading=false cache.use_cache_after_transfroms=false

# UNIBO with battery/unibo transforms
uv run python pseudo_run.py experiment=unibo/prognostics/base transforms=battery/unibo/combined_fit_predict cache.use_cache_after_transfroms=false

uv run python pseudo_run.py experiment=unibo/prognostics/base transforms=battery/unibo/ablation_missing_values_combined cache.use_cache_after_transfroms=false
