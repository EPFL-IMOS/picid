# # NB1
# python lmetk/run.py experiment=nb1/prognostics/stf debug=stf
# python lmetk/run.py experiment=nb1/prognostics/tabpfn_fit_predict debug=stf
# python lmetk/run.py experiment=nb1/prognostics/tabpfn_batch_context debug=stf
# # NB14 
# python lmetk/run.py experiment=nb14/prognostics/stf debug=stf
# python lmetk/run.py experiment=nb14/prognostics/tabpfn_batch_context debug=stf
# # ETTH
# python lmetk/run.py experiment=etth/multivariate/mean debug=stf
# python lmetk/run.py experiment=etth/multivariate/naive debug=stf
# python lmetk/run.py experiment=etth/multivariate/timepfn_batch_context debug=stf
# python lmetk/run.py experiment=etth/univariate/mean debug=stf
# python lmetk/run.py experiment=etth/univariate/naive debug=stf
# python lmetk/run.py experiment=etth/univariate/timepfn_batch_context debug=stf
# # RAILWAY
# python lmetk/run.py experiment=railway_traction/stf
# python lmetk/run.py experiment=railway_traction/tabpfn_no_target_context
# python lmetk/run.py experiment=railway_traction/tabpfn_railway_seasonal_context
# python lmetk/run.py experiment=railway_traction/timepfn
# python lmetk/run.py experiment=railway_traction/tabpfn_fit_predict


wandb_log_folder=unibo_prognostics
device=0
max_epochs=250
seq=64
python lmetk/run.py experiment=unibo/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
python lmetk/run.py experiment=unibo/prognostics/bosello23_base_satistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
wait

seq=16
python lmetk/run.py experiment=unibo/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
python lmetk/run.py experiment=unibo/prognostics/bosello23_base_satistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
wait

seq=32
python lmetk/run.py experiment=unibo/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
python lmetk/run.py experiment=unibo/prognostics/bosello23_base_satistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
wait


seq=128
python lmetk/run.py experiment=unibo/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
python lmetk/run.py experiment=unibo/prognostics/bosello23_base_satistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[0] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq &
wait