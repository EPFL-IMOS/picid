#!/bin/bash

# ==========================================================
# SETUP: Prepare the logs directory (keep past runs for resume)
# ==========================================================
LOG_DIR="./logs"
echo "Ensuring log directory exists at: $LOG_DIR (existing logs preserved)"
mkdir -p "$LOG_DIR"
# Clean up any stale FIFO pipes from previous runs
find "$LOG_DIR" -maxdepth 1 -type p -name "*.pipe.*" -delete 2>/dev/null || true

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

max_epochs=200
check_val_every_n_epoch=1
device=0
wandb_log_folder="16_03_2026"
# Set DEBUG_SKIP to any non-empty value to log runs without executing commands
DEBUG_SKIP="${DEBUG_SKIP:-}"
# Define a list of datasets to run

datasets=(
#   "mzvav||diagnostics"
  "hsf15/accumulator||diagnostics"
  "hsf15/cooler||diagnostics"
  "hsf15/pump||diagnostics"
  "hsf15/valve||diagnostics"
#   "concepts_n_cmapss_multi||diagnostics"
)

model_names=(
    # lstm
    # cnn_1d
    # mlp
    # linear_classifier
    # stf
    crossformer
    # timeseries_transformer
    # tide
    # patchtst
)

BATCH_SIZES=(512)
SEEDS=(72 88 101 666 226688)
SEQ_LEN=(10 50)
MAX_LEARNING_RATE=(0.001 0.0005 0.0001)

# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================

# --- 2. Initialize counters for tracking runs and managing parallel jobs ---
run_counter=1
job_counter=0
MAX_PARALLEL=1 # Set the max number of jobs to run at once

SUCCESS_LOG="${LOG_DIR}/SUCCESSFUL_RUNS.log"


build_exp_name() {
  local dataset_key="$1"
  local task_type="$2"
  local subexp="$3"
  local model_name="$4"

  if [[ -n "$subexp" ]]; then
    printf '%s/%s/%s/%s' "$dataset_key" "$task_type" "$subexp" "$model_name"
  else
    printf '%s/%s/%s' "$dataset_key" "$task_type" "$model_name"
  fi
}

pids=()
trap 'echo "Caught interrupt, killing jobs (${pids[*]})..."; kill "${pids[@]}" 2>/dev/null; exit 1' SIGINT SIGTERM


for dataset in "${datasets[@]}"; do
    for model_name in "${model_names[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for seq_len in "${SEQ_LEN[@]}"; do
                for max_learning_rate in "${MAX_LEARNING_RATE[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        # --- 3. Create a unique and descriptive name for this specific run ---
                        # Formatted with leading zeros (e.g., 001, 002) for proper sorting.

                        IFS='|' read -r dataset_key subexp task_type <<< "$dataset"
                        exp_name="$(build_exp_name "$dataset_key" "$task_type" "$subexp" "$model_name")"
                        dataset_run_name="${dataset_key//\//_}"
                        run_name=$(printf "%03d_%s_%s_%s_%s_%s" "$run_counter" "$dataset_run_name" "$model_name" "$task_type" "$subexp" "$seed" )
                        final_log_folder="${wandb_log_folder}_${dataset_run_name}"_"${task_type}"_"${subexp}"

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
                            experiment=${exp_name} \
                            datamodule.train_batch_size=${batch_size} \
                            datamodule.val_batch_size=1024 \
                            datamodule.test_batch_size=1024 \
                            task_definition.seq_len=${seq_len} \
                            optimization.lr=${max_learning_rate} \
                            trainer.check_val_every_n_epoch=${check_val_every_n_epoch} \
                            trainer.max_epochs=${max_epochs} \
                            trainer.devices=[${device}] \
                            logger.wandb.project=${final_log_folder} \
                            cache.use_cache_after_loading=True \
                            cache.use_cache_after_transfroms=True \
                            cache.use_preprocessing_file_lock=${use_preprocessing_file_lock} \
                            cache.preprocessing_file_lock_path=/tmp/picid_preprocess.lock \
                            enable_progress_bar=False \
                            seed=${seed}
                        )

                        final_command="${command_to_run[*]}"

                        success_key="${run_name}|${final_command}"

                        # Skip if this exact run/command pair already succeeded in a previous attempt
                        if [ -f "${SUCCESS_LOG}" ]; then
                            if grep -Fq "${success_key}" "${SUCCESS_LOG}"; then
                                echo "Skipping ${run_name} (command already recorded as successful)."
                                ((run_counter++))
                                continue
                            fi
                        fi

                        # Random sleep only for runs 2 and 3
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
                        run_and_log "${command_to_run[*]}" "$run_name" "$DEBUG_SKIP" &
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
done

# Wait for any remaining jobs to complete before exiting the script
echo "Waiting for the final batch of jobs to finish..."
wait
echo "All runs complete."
