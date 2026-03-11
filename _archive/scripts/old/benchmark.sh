# NB1
# python picid/run.py experiment=nb1/prognostics/stf debug=stf
# python picid/run.py experiment=nb1/prognostics/tabpfn_fit_predict debug=stf
# python picid/run.py experiment=nb1/prognostics/tabpfn_batch_context debug=stf

# NB14
# python picid/run.py experiment=nb14/prognostics/stf debug=stf
# python picid/run.py experiment=nb14/prognostics/tabpfn_batch_context debug=stf

# UNIBO

# NCMAPSS
python picid/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/cnn_1d
python picid/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/crossformer
python picid/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/lstm
python picid/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/patchtst
python picid/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/stf
python picid/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/tabpfn_fit_predict
python picid/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/tide
python picid/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/timeseries_transformer
python picid/run.py paths=rt_local experiment=concepts_n_cmapss/prognostics/xgboost_fit_predict
# NCMAPSS DS02
python picid/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/cnn_1d
python picid/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/crossformer
python picid/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/lstm
python picid/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/patchtst
python picid/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/stf
python picid/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/tabpfn_fit_predict
python picid/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/tide
python picid/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/timeseries_transformer
python picid/run.py paths=rt_local experiment=concepts_n_cmapss_ds02/prognostics/xgboost_fit_predict

# XJTU-SY

# ETTH
# python picid/run.py experiment=etth/multivariate/mean debug=stf
# python picid/run.py experiment=etth/multivariate/naive debug=stf
# python picid/run.py experiment=etth/multivariate/timepfn_batch_context debug=stf

# python picid/run.py experiment=etth/univariate/mean debug=stf
# python picid/run.py experiment=etth/univariate/naive debug=stf
# python picid/run.py experiment=etth/univariate/timepfn_batch_context debug=stf

# RAILWAY
python picid/run.py paths=rt_local experiment=railway_traction/forecasting/tabpfn_fit_predict

# python picid/run.py experiment=railway_traction/stf
# python picid/run.py experiment=railway_traction/tabpfn_no_target_context
# python picid/run.py experiment=railway_traction/tabpfn_railway_seasonal_context
# python picid/run.py experiment=railway_traction/timepfn
# python picid/run.py experiment=railway_traction/tabpfn_fit_predict
# # NB1
# python picid/run.py experiment=nb1/prognostics/stf debug=stf
# python picid/run.py experiment=nb1/prognostics/tabpfn_fit_predict debug=stf
# python picid/run.py experiment=nb1/prognostics/tabpfn_batch_context debug=stf
# # NB14
# python picid/run.py experiment=nb14/prognostics/stf debug=stf
# python picid/run.py experiment=nb14/prognostics/tabpfn_batch_context debug=stf
# # ETTH
# python picid/run.py experiment=etth/multivariate/mean debug=stf
# python picid/run.py experiment=etth/multivariate/naive debug=stf
# python picid/run.py experiment=etth/multivariate/timepfn_batch_context debug=stf
# python picid/run.py experiment=etth/univariate/mean debug=stf
# python picid/run.py experiment=etth/univariate/naive debug=stf
# python picid/run.py experiment=etth/univariate/timepfn_batch_context debug=stf
# # RAILWAY
# python picid/run.py experiment=railway_traction/stf
# python picid/run.py experiment=railway_traction/tabpfn_no_target_context
# python picid/run.py experiment=railway_traction/tabpfn_railway_seasonal_context
# python picid/run.py experiment=railway_traction/timepfn
# python picid/run.py experiment=railway_traction/tabpfn_fit_predict


wandb_log_folder=unibo_prognostics
device=0
max_epochs=250
seq=64
python picid/run.py experiment=unibo/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
python picid/run.py experiment=unibo/prognostics/bosello23_base_satistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
wait

seq=16
python picid/run.py experiment=unibo/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
python picid/run.py experiment=unibo/prognostics/bosello23_base_satistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
wait

seq=32
python picid/run.py experiment=unibo/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
python picid/run.py experiment=unibo/prognostics/bosello23_base_satistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
wait


seq=128
python picid/run.py experiment=unibo/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
python picid/run.py experiment=unibo/prognostics/bosello23_base_satistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
wait
