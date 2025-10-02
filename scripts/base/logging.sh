# File: run_utils.sh

#!/bin/bash

# This function executes a command, logs its output, and immediately
# records its success or failure to dedicated log files.
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

    # Execute the command
    # The >/dev/null for stdout here cleans up the terminal, as we only care about errors on failure.
    eval "$cmd" > >(tee -a "$stdout_log") 2> >(tee -a "$stderr_log" >&2)

    # Check the exit status of the 'eval' command. 0 means success.
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ [SUCCESS] Finished: $run_name"
        echo "$(date): ${run_name}" >> "$success_log"
        return 0
    else
        echo "❌ [FAILURE] Finished: $run_name"
        
        # ✨ NEW: Print the last 15 lines of the error log to the console
        echo "----------------- ERROR OUTPUT -----------------"
        tail -n 15 "$stderr_log"
        echo "------------------------------------------------"

        # This part logs the failure for your records, as before
        echo "=================================" >> "$failed_log"
        echo "FAILURE on $(date): $run_name" >> "$failed_log"
        echo "Command: $cmd" >> "$failed_log"
        echo "See full logs: $stdout_log | $stderr_log" >> "$failed_log"
        echo "=================================" >> "$failed_log"
        return 1
    fi
}