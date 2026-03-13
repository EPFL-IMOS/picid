wandb_log_folder=pronostia_prognostics
device=7
max_epochs=100

seq=1024
python picid/run.py experiment=pronostia/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[$device] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq  task_definition.stride=$seq &
# device=4
# python picid/run.py experiment=pronostia/prognostics/statistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[$device] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq  task_definition.stride=$seq &
wait







seq=2048
python picid/run.py experiment=pronostia/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[$device] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq  task_definition.stride=$seq &

# device=4
# python picid/run.py experiment=pronostia/prognostics/statistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[$device] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq  task_definition.stride=$seq &
wait


seq=512
python picid/run.py experiment=pronostia/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[$device] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq  task_definition.stride=$seq &
# device=4
# python picid/run.py experiment=pronostia/prognostics/statistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[$device] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq  task_definition.stride=$seq &
wait
