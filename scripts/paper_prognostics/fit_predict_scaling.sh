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

check_val_every_n_epoch=1
device=0
wandb_log_folder="16_04_2026_scaling"
# Set DEBUG_SKIP to any non-empty value to log runs without executing commands
DEBUG_SKIP="${DEBUG_SKIP:-}"
# Define a list of datasets to run

datasets=(
#   "nb14|raw|prognostics"
#   "nb14|combined|prognostics"
  "unibo|combined|prognostics" # TODO
#   "phme20|raw|prognostics"
#   "concepts_n_cmapss||prognostics"
#   "concepts_n_cmapss_ds02||prognostics"  # TODO
#   "pronostia|raw|prognostics"
#   "pronostia|combined|prognostics"
#   "xjtu_sy|combined|prognostics"
)

subset_ratios=(0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
SEEDS=(72 88 101 666 226688)

model_names=(
    # xgboost_fit_predict
    tabdpt_fit_predict
    tabpfn_fit_predict
)

# Careful here because of oom issues with long context lengths / short strides
# Pairs: "seq_len:stride_train"
context_stride_pairs=(
#   "1:1"
#   "5:1"
#   "10:5"
  "20:5"
#   "50:50"
#   "100:100"
)

# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================

# --- 2. Initialize counters for tracking runs and managing parallel jobs ---
run_counter=1
job_counter=0
MAX_PARALLEL=1 # Set the max number of jobs to run at once

SUCCESS_LOG="${LOG_DIR}/SUCCESSFUL_RUNS.log"

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


for seed in "${SEEDS[@]}"; do
    for dataset in "${datasets[@]}"; do
        for model_name in "${model_names[@]}"; do
            for ratio in "${subset_ratios[@]}"; do
                for cs_pair in "${context_stride_pairs[@]}"; do
                    # --- 3. Create a unique and descriptive name for this specific run ---
                    # Formatted with leading zeros (e.g., 001, 002) for proper sorting.
                    IFS=':' read -r context_length train_set_stride <<< "$cs_pair"
                    IFS='|' read -r dataset_key subexp task_type <<< "$dataset"
                    exp_name="$(build_exp_name "$dataset_key" "$task_type" "$subexp" "$model_name")"
                    dataset_run_name="${dataset_key//\//_}"
                    run_name=$(printf "%03d_%s_%s_%s_%s_%s" "$run_counter" "$dataset_run_name" "$model_name" "$task_type" "$subexp" "$seed" )
                    final_log_folder="${wandb_log_folder}_${dataset_run_name}"_"${task_type}"_"${subexp}"

                    # --- 4. Build the full command as a single string ---
                    # Note the escaped quotes \"...\" for parameters that are lists.
                    command_to_run=(
                        python picid/run.py \
                        paths=runai \
                        experiment=${exp_name} \
                        logger.wandb.project=${final_log_folder} \
                        # task_definition.stride_train=${train_set_stride} \
                        cache.use_cache_after_loading=True \
                        cache.use_cache_after_transfroms=True \
                        task_definition.subset_ratio=${ratio} \
                        task_definition.seq_len=${context_length} \
                        experiment_group=${ratio} \
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

# Wait for any remaining jobs to complete before exiting the script
echo "Waiting for the final batch of jobs to finish..."
wait
echo "All runs complete."
