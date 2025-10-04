# NB1
# python lmetk/run.py experiment=nb1/prognostics/stf debug=stf
# python lmetk/run.py experiment=nb1/prognostics/tabpfn_fit_predict debug=stf
# python lmetk/run.py experiment=nb1/prognostics/tabpfn_batch_context debug=stf

# NB14 
# python lmetk/run.py experiment=nb14/prognostics/stf debug=stf
# python lmetk/run.py experiment=nb14/prognostics/tabpfn_batch_context debug=stf

# UNIBO

# NCMAPSS
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/cnn_1d
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/crossformer
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/lstm
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/patchtst
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/stf
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/tabpfn_fit_predict
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/tide
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/timeseries_transformer
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/xgboost_fit_predict
# NCMAPSS DS02
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/cnn_1d
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/crossformer
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/lstm
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/patchtst
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/stf
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/tabpfn_fit_predict
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/tide
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/timeseries_transformer
python lmetk/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/xgboost_fit_predict 

# XJTU-SY

# ETTH
# python lmetk/run.py experiment=etth/multivariate/mean debug=stf
# python lmetk/run.py experiment=etth/multivariate/naive debug=stf
# python lmetk/run.py experiment=etth/multivariate/timepfn_batch_context debug=stf

# python lmetk/run.py experiment=etth/univariate/mean debug=stf
# python lmetk/run.py experiment=etth/univariate/naive debug=stf
# python lmetk/run.py experiment=etth/univariate/timepfn_batch_context debug=stf

# RAILWAY
python lmetk/run.py paths=rt_local experiment=railway_traction/forecasting/tabpfn_fit_predict

# python lmetk/run.py experiment=railway_traction/stf
# python lmetk/run.py experiment=railway_traction/tabpfn_no_target_context
# python lmetk/run.py experiment=railway_traction/tabpfn_railway_seasonal_context
# python lmetk/run.py experiment=railway_traction/timepfn
# python lmetk/run.py experiment=railway_traction/tabpfn_fit_predict