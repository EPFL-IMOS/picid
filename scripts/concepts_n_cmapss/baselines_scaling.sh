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
source "$SCRIPT_DIR/../base/logging.sh"



max_epochs=50
check_val_every_n_epoch=1
device=0
wandb_log_folder="n_cmapss_scaling"
# Define a list of datasets to run
datasets=(concepts_n_cmapss_ds02 concepts_n_cmapss)
subset_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
model_names=(crossformer lstm patchtst stf tide timeseries_transformer)
BATCH_SIZES=(512)
SEEDS=(72 88 101 666 226688)

# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================

# --- 2. Initialize counters for tracking runs and managing parallel jobs ---
run_counter=1
job_counter=0
MAX_PARALLEL=3 # Set the max number of jobs to run at once

pids=()
trap 'echo "Caught interrupt, killing jobs (${pids[*]})..."; kill "${pids[@]}" 2>/dev/null; exit 1' SIGINT SIGTERM


for model_name in "${model_names[@]}"; do
    for dataset in "${datasets[@]}"; do
        for ratio in "${subset_ratios[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    # --- 3. Create a unique and descriptive name for this specific run ---
                    # Formatted with leading zeros (e.g., 001, 002) for proper sorting.
                    run_name=$(printf "%03d_%s_%s" "$run_counter" "$dataset" "$seed" )

                    # --- 4. Build the full command as a single string ---
                    # Note the escaped quotes \"...\" for parameters that are lists.

                    # Use file lock only for the first run to avoid deadlocks once cache is built
                    if (( run_counter == 1 )); then
                        use_preprocessing_file_lock=True
                    else
                        use_preprocessing_file_lock=False
                    fi

                    command_to_run=(
                        python lmetk/run.py \
                        paths=runai \
                        experiment=${dataset}/prognostics/${model_name} \
                        datamodule.train_batch_size=${batch_size} \
                        datamodule.val_batch_size=1024 \
                        datamodule.test_batch_size=1024 \
                        trainer.check_val_every_n_epoch=${check_val_every_n_epoch} \
                        trainer.max_epochs=${max_epochs} \
                        trainer.devices=[${device}] \
                        logger.wandb.project='${wandb_log_folder}' \
                        cache.use_cache_after_loading=True \
                        cache.use_cache_after_transfroms=True \
                        cache.use_preprocessing_file_lock=${use_preprocessing_file_lock} \
                        cache.preprocessing_file_lock_path=/tmp/lmetk_preprocess.lock \
                        task_definition.subset_ratio=${ratio} \
                        enable_progress_bar=False \
                        experiment_group=${ratio} \
                        seed=${seed}
                    )

                    # Random sleep only for runs 2, 3 up to MAX_PARALLEL to stagger preprocessing starts
                    if (( run_counter > 1 && run_counter <= $MAX_PARALLEL )); then
                        sleep_time=$((60 * (RANDOM % 3 + 1)))  # 1–3 minutes
                        echo "Delaying launch of run ${run_name} by ${sleep_time}s..."
                        sleep $sleep_time
                    fi

                    # Block all processes except the first one here to prevent multiple preprocessings at once
                    if (( run_counter > 1 )); then
                    (
                        flock 200
                    ) 200>/tmp/lmetk_preprocess.lock
                    echo "Python lock released, continuing..."
                    fi

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
done

# Wait for any remaining jobs to complete before exiting the script
echo "Waiting for the final batch of jobs to finish..."
wait
echo "All runs complete."