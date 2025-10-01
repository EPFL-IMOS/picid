# File: run_utils.sh

#!/bin/bash

# This function executes a command and logs its outcome.
# It saves detailed stdout/stderr logs ONLY for failed runs.
run_and_log() {
    local cmd="$1"
    local run_name="$2"

    # Define log file paths
    local log_dir="./logs"
    mkdir -p "$log_dir"
    local stdout_log="$log_dir/${run_name}_stdout.log"
    local stderr_log="$log_dir/${run_name}_stderr.log"
    local success_log="$log_dir/SUCCESSFUL_RUNS.log"
    local failed_log="$log_dir/FAILED_RUNS.log"

    echo "--- [START] Running: $run_name ---"

    # Execute the command, temporarily creating full logs for stdout and stderr
    eval "$cmd" > >(tee -a "$stdout_log") 2> >(tee -a "$stderr_log" >&2)

    # Check the exit status of the command
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        # --- SUCCESS CASE ---
        echo "✅ [SUCCESS] Finished: $run_name"
        # 1. Log the success to the summary file
        echo "$(date): ${run_name}" >> "$success_log"

        # 2. ✅ Clean up the detailed logs as they are no longer needed
        rm -f "$stdout_log" "$stderr_log"
        
        return 0
    else
        # --- FAILURE CASE ---
        echo "❌ [FAILURE] Finished: $run_name. Keeping full logs for review."
        # 1. Log the failure and details to the summary file
        echo "=================================" >> "$failed_log"
        echo "FAILURE on $(date): $run_name" >> "$failed_log"
        echo "Command: $cmd" >> "$failed_log"
        echo "See logs: $stdout_log | $stderr_log" >> "$failed_log"
        echo "=================================" >> "$failed_log"
        
        # 2. The detailed logs are automatically kept.
        return 1
    fi
}