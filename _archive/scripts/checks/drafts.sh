python picid/run.py experiment=pronostia/prognostics/spectral/cnn_1d trainer.max_epochs=150 trainer.min_epochs=100 optimizer.parameters.lr=0.001 task_definition.seq_len=64 evaluator.train.apply_inverse_scaling=False evaluator.val.apply_inverse_scaling=False evaluator.test.apply_inverse_scaling=False


python picid/run.py experiment=pronostia/prognostics/spectral/cnn_1d trainer.max_epochs=1000 trainer.min_epochs=500 optimizer.parameters.lr=0.001 task_definition.seq_len=64 evaluator.train.apply_inverse_scaling=False evaluator.val.apply_inverse_scaling=False evaluator.test.apply_inverse_scaling=False datamodule.num_workers=0 datamodule.val_batch_size=1024
