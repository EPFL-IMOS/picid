#!/bin/bash

# Reproduce best run from report (same best HPs, all seeds).
# Dataset: UNIBO21 -> unibo|combined|prognostics, Model: model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper -> tabdpt_fit_predict

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Source logging from project scripts (relative to scripts/reproduce -> scripts/base)
source "$SCRIPT_DIR/../base/logging.sh"

export HYDRA_FULL_ERROR=1
wandb_log_folder="reproduce_29_01_2026"
DEBUG_SKIP="${DEBUG_SKIP:-}"

dataset="unibo|combined|prognostics"
model_name="tabdpt_fit_predict"
exp_name="unibo/prognostics/combined/tabdpt_fit_predict"
final_log_folder="reproduce_29_01_2026_unibo_prognostics_combined"
dataset_run_name="unibo"

run_name="001_unibo_tabdpt_fit_predict_prognostics_combined_72"

command_to_run=(
    python picid/run.py \
    paths=runai \
    experiment=unibo/prognostics/combined/tabdpt_fit_predict \
    logger.wandb.project=reproduce_29_01_2026_unibo_prognostics_combined \
    task_definition.seq_len=1 \
    task_definition.stride_train=1 \
    cache.use_cache_after_loading=True \
    cache.use_cache_after_transfroms=True
)

run_and_log "${command_to_run[*]}" "$run_name" "$DEBUG_SKIP"

