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
echo "Script directory determined as: $SCRIPT_DIR"

# Add this line to get detailed errors from Hydra
export HYDRA_FULL_ERROR=1
# Source the logging.sh file using the script's directory as an anchor.
source "$SCRIPT_DIR/base/logging.sh"



max_epochs=20
check_val_every_n_epoch=1
device=0
wandb_log_folder="TabPFN_ds02_sweep_2025_10_09"
# Define a list of datasets to run
datasets=(concepts_n_cmapss_ds02)
model_names=(cnn_1d crossformer lstm patchtst stf tide timeseries_transformer xgboost_fit_predict)
BATCH_SIZES=(16 512)
SEEDS=(72 88 101 666 226688)

# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================

# --- 2. Initialize counters for tracking runs and managing parallel jobs ---
run_counter=1
job_counter=0
MAX_PARALLEL=3 # Set the max number of jobs to run at once


# Function to run and log commands
# run_and_log() {
#     local command_to_run="$1"
#     local run_name="$2"

#     # For debugging, just print the command and run name
#     echo "DEBUG: Run Name: $run_name"
#     echo "DEBUG: Command: $command_to_run"
#     # Run the command and log its output
#     eval "sleep 5"
# }

pids=()
trap 'echo "Caught interrupt, killing jobs (${pids[*]})..."; kill "${pids[@]}" 2>/dev/null; exit 1' SIGINT SIGTERM


for model_name in "${model_names[@]}"; do
    for dataset in "${datasets[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                # --- 3. Create a unique and descriptive name for this specific run ---
                # Formatted with leading zeros (e.g., 001, 002) for proper sorting.
                run_name=$(printf "%03d_%s_%s" "$run_counter" "$dataset" "$seed" )

                # --- 4. Build the full command as a single string ---
                # Note the escaped quotes \"...\" for parameters that are lists.
                command_to_run=(
                    python lmetk/run.py \
                    paths=runai \
                    experiment=${dataset}/prognostics/${model_name} \
                    datamodule.train_batch_size=${batch_size} \
                    datamodule.val_batch_size=1024 \
                    datamodule.test_batch_size=1024 \
                    optimizer.parameters.weight_decay=0.0 \
                    trainer.check_val_every_n_epoch=${check_val_every_n_epoch} \
                    trainer.max_epochs=${max_epochs} \
                    trainer.devices=[${device}] \
                    logger.wandb.project='${wandb_log_folder}' \
                    cache.use_cache_after_loading=True \
                    cache.use_cache_after_transfroms=True
                    seed=${seed}
                )
                # debug=train_debug
                # --- 5. Execute using the logger and run in the background (&) ---
                run_and_log "${command_to_run[*]}" "$run_name" &
                pid=$!
                pids+=($pid)
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

# Wait for any remaining jobs to complete before exiting the script
echo "Waiting for the final batch of jobs to finish..."
wait
echo "All runs complete."