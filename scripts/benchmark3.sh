wandb_log_folder=unibo_prognostics_like_bosello_paper
device=3
max_epochs=250
seq=1
python lmetk/run.py experiment=unibo/prognostics/autoencoder/stf logger.wandb.project=$wandb_log_folder trainer.devices=[2] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq evaluator.train.apply_inverse_scaling=False evaluator.val.apply_inverse_scaling=False evaluator.test.apply_inverse_scaling=False &
python lmetk/run.py experiment=unibo/prognostics/bosello23_base_satistics/stf logger.wandb.project=$wandb_log_folder trainer.devices=[2] trainer.max_epochs=$max_epochs task_definition.seq_len=$seq evaluator.train.apply_inverse_scaling=False evaluator.val.apply_inverse_scaling=False evaluator.test.apply_inverse_scaling=False &

