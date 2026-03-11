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



max_epochs=1
device=0
wandb_log_folder="unibo_full"
# Define a list of datasets to run
datasets=(unibo)
subset_ratios=(1.0)
model_names=(tabdpt_fit_predict tabpfn_fit_predict xgboost_fit_predict)
SEEDS=(72 88 101 666 226688)
context_lengths=(1 5)

# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================

# --- 2. Initialize counters for tracking runs and managing parallel jobs ---
run_counter=1
job_counter=0
MAX_PARALLEL=1 # Set the max number of jobs to run at once

pids=()
trap 'echo "Caught interrupt, killing jobs (${pids[*]})..."; kill "${pids[@]}" 2>/dev/null; exit 1' SIGINT SIGTERM


for model_name in "${model_names[@]}"; do
    for dataset in "${datasets[@]}"; do
        for ratio in "${subset_ratios[@]}"; do
            for context_length in "${context_lengths[@]}"; do
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
                        python picid/run.py \
                        paths=runai \
                        experiment=${dataset}/prognostics/statistics/${model_name} \
                        logger.wandb.project='${wandb_log_folder}' \
                        cache.use_cache_after_loading=True \
                        cache.use_cache_after_transfroms=True \
                        cache.use_preprocessing_file_lock=${use_preprocessing_file_lock} \
                        cache.preprocessing_file_lock_path=/tmp/picid_preprocess.lock \
                        task_definition.subset_ratio=${ratio} \
                        task_definition.seq_len=${context_length} \
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
                    ) 200>/tmp/picid_preprocess.lock
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
