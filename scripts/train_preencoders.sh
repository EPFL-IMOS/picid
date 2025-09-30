


python lmetk/run.py paths=rt_local  experiment=nb14/reconstruction/residual_autoencoder trainer.max_epochs=2 trainer.check_val_every_n_epoch=1
python lmetk/run.py paths=rt_local  experiment=unibo/reconstruction/residual_autoencoder trainer.max_epochs=2 trainer.check_val_every_n_epoch=1
python lmetk/run.py paths=rt_local  experiment=xjtu_sy/reconstruction/residual_autoencoder trainer.max_epochs=2 trainer.check_val_every_n_epoch=1
python lmetk/run.py paths=rt_local  experiment=pronostia/reconstruction/residual_autoencoder trainer.max_epochs=2 trainer.check_val_every_n_epoch=1