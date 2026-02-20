#!/bin/bash

# Reproduce best run from report (same best HPs, all seeds).
# Dataset: UNIBO21 -> unibo|combined|prognostics, Model: model.wrappers.mlp_wrapper.MLPWrapper -> mlp

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Source logging from project scripts (relative to scripts/reproduce -> scripts/base)
source "$SCRIPT_DIR/../base/logging.sh"

export HYDRA_FULL_ERROR=1
wandb_log_folder="reproduce_29_01_2026"
DEBUG_SKIP="${DEBUG_SKIP:-}"

dataset="unibo|combined|prognostics"
model_name="mlp"
exp_name="unibo/prognostics/combined/mlp"
final_log_folder="reproduce_29_01_2026_unibo_prognostics_combined"
dataset_run_name="unibo"

SEEDS=(72 88 101 666 226688)

for seed in "${SEEDS[@]}"; do
  run_name="001_unibo_mlp_prognostics_combined_${seed}"

  command_to_run=(
    python picid/run.py \
    paths=runai \
    experiment=unibo/prognostics/combined/mlp \
    datamodule.train_batch_size=512 \
    datamodule.val_batch_size=1024 \
    datamodule.test_batch_size=1024 \
    task_definition.seq_len=50 \
    optimization.lr=0.001 \
    trainer.check_val_every_n_epoch=1 \
    trainer.max_epochs=200 \
    trainer.devices=[0] \
    logger.wandb.project=reproduce_29_01_2026_unibo_prognostics_combined \
    cache.use_cache_after_loading=True \
    cache.use_cache_after_transfroms=True \
    cache.use_preprocessing_file_lock=False \
    cache.preprocessing_file_lock_path=/tmp/picid_preprocess.lock \
    enable_progress_bar=False \
    seed=${seed}
  )

  run_and_log "${command_to_run[*]}" "$run_name" "$DEBUG_SKIP"
done

