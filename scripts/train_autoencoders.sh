#!/bin/bash

# ==========================================================
# SETUP: Clean and prepare the logs directory for a fresh run
# ==========================================================
LOG_DIR="./logs"
echo "Preparing a clean log directory at: $LOG_DIR"

# If the log directory exists, delete it and everything inside it
if [ -d "$LOG_DIR" ]; then
    rm -r "$LOG_DIR"
fi

# Create a new, empty log directory
mkdir -p "$LOG_DIR"

# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================
# Get the absolute path to the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Add this line to get detailed errors from Hydra
export HYDRA_FULL_ERROR=1
# Source the logging.sh file using the script's directory as an anchor.
source "$SCRIPT_DIR/base/logging.sh"



max_epochs=20
check_val_every_n_epoch=1
device=1
wandb_log_folder="TabPHM-autoencoders-checks_02_10_night"
# Define a list of datasets to run
datasets=(unibo pronostia) # unibo  xjtu_sy nb14
BATCH_SIZES=(128 256 512 1024)

# ========================================================================
# ✅ CHANGE 1: Define each model's parameters in an associative array.
# This groups all related parameter lists under a single model name.
# ========================================================================
declare -A MODEL_STRIDE_RF_67
MODEL_STRIDE_RF_67[output_channels]="[16,32,64,64,64]"
MODEL_STRIDE_RF_67[kernels]="[7,7,5,5,5]"
MODEL_STRIDE_RF_67[strides]="[2,2,1,1,1]"
MODEL_STRIDE_RF_67[dilations]="[1,1,1,1,1]"

declare -A MODEL_DILATED_RF_65
MODEL_DILATED_RF_65[output_channels]="[16,32,64,64,64]"
MODEL_DILATED_RF_65[kernels]="[5,5,5,5,5]"
MODEL_DILATED_RF_65[strides]="[1,1,1,1,1]"
MODEL_DILATED_RF_65[dilations]="[1,2,4,8,1]"

declare -A MODEL_TEN_LAYER_RF_64
MODEL_TEN_LAYER_RF_64[output_channels]="[16,32,64,64,64,16,32,64,64,64]"
MODEL_TEN_LAYER_RF_64[kernels]="[10,7,7,7,7,7,7,7,7,7]"
MODEL_TEN_LAYER_RF_64[strides]="[1,1,1,1,1,1,1,1,1,1]"
MODEL_TEN_LAYER_RF_64[dilations]="[1,1,1,1,1,1,1,1,1,1]"


# ========================================================================
# ✅ CHANGE 2: Store the NAMES of the associative arrays.
# The main loop will use these names to access the full parameter sets.
# ========================================================================
ARCHS_FOR_SEQ_LEN_64=(
  MODEL_STRIDE_RF_67
  MODEL_DILATED_RF_65
  MODEL_TEN_LAYER_RF_64
)
# Define ARCHS_FOR_SEQ_LEN_128=(...) etc. here as needed


# ========================
# Define valid seq_lens per dataset
# ========================
declare -A DATASET_SEQ_MAP
DATASET_SEQ_MAP["unibo"]="64"
DATASET_SEQ_MAP["nb14"]="64 128" # Example with multiple lengths
DATASET_SEQ_MAP["pronostia"]="64"
DATASET_SEQ_MAP["xjtu_sy"]="64"

# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================

# --- 2. Initialize counters for tracking runs and managing parallel jobs ---
run_counter=1
job_counter=0
MAX_PARALLEL=2 # Set the max number of jobs to run at once

for dataset in "${!DATASET_SEQ_MAP[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for lr in 0.01 0.001 0.0005; do
            for step_size in 10 20; do
                for seq in ${DATASET_SEQ_MAP[$dataset]}; do
                
                    declare -n arch_list="ARCHS_FOR_SEQ_LEN_${seq}"
                    for arch_name in "${arch_list[@]}"; do
                        declare -n current_model_params="$arch_name"

                        # --- 3. Create a unique and descriptive name for this specific run ---
                        # Formatted with leading zeros (e.g., 001, 002) for proper sorting.
                        run_name=$(printf "%03d_%s_seq%s_%s" "$run_counter" "$dataset" "$seq" "$arch_name")

                        # --- 4. Build the full command as a single string ---
                        # Note the escaped quotes \"...\" for parameters that are lists.
                        command_to_run="python lmetk/run.py \
                        experiment=${dataset}/reconstruction/residual_autoencoder \
                        task_definition.seq_len=$seq \
                        datamodule.train_batch_size=${batch_size} \
                        datamodule.val_batch_size=1024 \
                        datamodule.test_batch_size=1024 \
                        model.output_channels=\"${current_model_params[output_channels]}\" \
                        model.kernels=\"${current_model_params[kernels]}\" \
                        model.strides=\"${current_model_params[strides]}\" \
                        model.dilations=\"${current_model_params[dilations]}\" \
                        optimizer.parameters.lr=${lr} \
                        optimizer.scheduler.scheduler_params.step_size=${step_size} \
                        optimizer.parameters.weight_decay=0.0 \
                        trainer.check_val_every_n_epoch=${check_val_every_n_epoch} \
                        trainer.max_epochs=${max_epochs} \
                        trainer.devices=[${device}] \
                        logger.wandb.project='${wandb_log_folder}' \
                        cache.use_cache_after_loading=True \
                        cache.use_cache_after_transfroms=True"
                        # debug=train_debug"
                        # --- 5. Execute using the logger and run in the background (&) ---
                        run_and_log "$command_to_run" "$run_name" &

                        # --- 6. Increment counters and manage parallel jobs ---
                        ((run_counter++))
                        ((job_counter++))
                        # First, check if all our "slots" are full.
                        if [[ "$job_counter" -ge "$MAX_PARALLEL" ]]; then
                            # If they are, wait for any single job to finish.
                            wait -n
                            # Decrement the counter because a slot just opened up.
                            ((job_counter--))
                        fi
                        
                    done
                done
            done
        done
    done
done


# Wait for any remaining jobs to complete before exiting the script
echo "Waiting for the final batch of jobs to finish..."
wait
echo "All runs complete."