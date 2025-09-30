#!/bin/bash
max_epochs=50
check_val_every_n_epoch=1
# Define a list of datasets to run
datasets=(nb14 unibo xjtu_sy pronostia)
BATCH_SIZES=(1024)
ARCH_SHALLOW_VAL='[{"in": 2, "out": 32, "kernel": 7, "stride": 2}, {"in": 32, "out": 64, "kernel": 5, "stride": 2}]'
ARCH_DEEP_VAL='[{"in": 2, "out": 16, "kernel": 7, "stride": 2}, {"in": 16, "out": 32, "kernel": 5, "stride": 2}, {"in": 32, "out": 64, "kernel": 3, "stride": 2}, {"in": 64, "out": 128, "kernel": 3, "stride": 2}]'
ARCH_WIDE_VAL='[{"in": 2, "out": 64, "kernel": 5, "stride": 2}, {"in": 64, "out": 256, "kernel": 3, "stride": 2}]'

# Create an array of these new value-only strings
ARCHITECTURES=(
  "$ARCH_SHALLOW_VAL"
  "$ARCH_DEEP_VAL"
  "$ARCH_WIDE_VAL"
)
# Loop through each dataset in the list
for dataset in "${datasets[@]}"; do
for batch_size in "${BATCH_SIZES[@]}"; do
for i in "${!ARCHITECTURES[@]}"; do
  echo "--- Running experiment for dataset: ${dataset} ---"
  # Get the full configuration string for the current iteration
  arch_config="${ARCHITECTURES[$i]}"
# DONT FORGET LATER: paths=rt_local \
  python lmetk/run.py \
    experiment=${dataset}/reconstruction/residual_autoencoder \
    trainer.max_epochs=${max_epochs} \
    trainer.check_val_every_n_epoch=${check_val_every_n_epoch} \
    datamodule.train_batch_size=${batch_size} \
    datamodule.val_batch_size=${batch_size} \
    datamodule.test_batch_size=${batch_size} \
    trainer.devices=[1] \
    model.blocks_config=${arch_config_val}
done
done
done
echo "--- All experiments finished. ---"



# python lmetk/run.py paths=rt_local  experiment=nb14/reconstruction/residual_autoencoder trainer.max_epochs=500 trainer.check_val_every_n_epoch=5
# python lmetk/run.py paths=rt_local  experiment=unibo/reconstruction/residual_autoencoder trainer.max_epochs=500 trainer.check_val_every_n_epoch=5
# python lmetk/run.py paths=rt_local  experiment=xjtu_sy/reconstruction/residual_autoencoder trainer.max_epochs=500 trainer.check_val_every_n_epoch=5
# python lmetk/run.py paths=rt_local  experiment=pronostia/reconstruction/residual_autoencoder trainer.max_epochs=500 trainer.check_val_every_n_epoch=5