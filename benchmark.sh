# NB1
python lmetk/run.py experiment=nb1/prognostics/stf debug=stf
python lmetk/run.py experiment=nb1/prognostics/tabpfn_fit_predict debug=stf
python lmetk/run.py experiment=nb1/prognostics/tabpfn_batch_context debug=stf


# NB14 
python lmetk/run.py experiment=nb14/prognostics/stf debug=stf
python lmetk/run.py experiment=nb14/prognostics/tabpfn_batch_context debug=stf


# ETTH
python lmetk/run.py experiment=etth/multivariate/mean debug=stf
python lmetk/run.py experiment=etth/multivariate/naive debug=stf
python lmetk/run.py experiment=etth/multivariate/timepfn_batch_context debug=stf

python lmetk/run.py experiment=etth/univariate/mean debug=stf
python lmetk/run.py experiment=etth/univariate/naive debug=stf
python lmetk/run.py experiment=etth/univariate/timepfn_batch_context debug=stf