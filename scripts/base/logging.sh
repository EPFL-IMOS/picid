# File: run_utils.sh

#!/bin/bash

# This function executes a command, logs its output, and keeps the logs
# ONLY if the command fails. Optional 3rd arg enables a debug "dry run"
# that records a SKIPPED entry while still exercising the logging path
# (pipes, tees, archiving) but skipping the actual command execution.
run_and_log() {
    local cmd="$1"
    local run_name="$2"
    local debug_skip="${3:-}"  # non-empty → do not execute command
    local skipped=false

    # Define log file paths
    local log_dir="./logs"
    mkdir -p "$log_dir"
    local file_run_name="${run_name//\//_}"
    file_run_name="${file_run_name// /_}"
    local stdout_log="$log_dir/${file_run_name}_stdout.log"
    local stderr_log="$log_dir/${file_run_name}_stderr.log"
    local success_log="$log_dir/SUCCESSFUL_RUNS.log"
    local failed_log="$log_dir/FAILED_RUNS.log"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"

    echo "--- [START] Running: $run_name ---"

    # Execute the command with robust streaming capture (avoids /dev/fd issues)
    local stdout_pipe
    local stderr_pipe
    stdout_pipe="$(mktemp -u "${stdout_log}.pipe.XXXX")"
    stderr_pipe="$(mktemp -u "${stderr_log}.pipe.XXXX")"
    mkfifo "$stdout_pipe" "$stderr_pipe"
    cleanup_pipes() { rm -f "$stdout_pipe" "$stderr_pipe"; }
    trap 'cleanup_pipes' RETURN
    trap 'cleanup_pipes; exit 1' INT TERM

    tee "$stdout_log" <"$stdout_pipe" &
    local tee_out_pid=$!
    tee "$stderr_log" <"$stderr_pipe" >&2 &
    local tee_err_pid=$!

    if [ -n "$debug_skip" ]; then
        echo "⚠️ [SKIPPED] Debug flag set; not running: $run_name" >"$stdout_pipe"
        # With probability 0.1 simulate a failure to exercise failure handling
        if (( RANDOM % 10 == 0 )); then
            echo "Simulated failure (debug skip mode)" >"$stderr_pipe"
            cmd_status=1
        else
            : >"$stderr_pipe"
            cmd_status=0
        fi
        skipped=true
    else
        eval "$cmd" >"$stdout_pipe" 2>"$stderr_pipe"
        cmd_status=$?
    fi

    # Close pipes and wait for tees to finish writing
    wait "$tee_out_pid"
    wait "$tee_err_pid"
    rm -f "$stdout_pipe" "$stderr_pipe"
    # Extra safety: remove any lingering FIFOs for this run name
    find "$log_dir" -maxdepth 1 -type p -name "${file_run_name}_*.pipe.*" -delete 2>/dev/null

    # Check the exit status of the command
    if [ $cmd_status -eq 0 ]; then
        if $skipped; then
            echo "⚠️ [SKIPPED] Finished: $run_name"
        else
            echo "✅ [SUCCESS] Finished: $run_name"
        fi
        # On success, archive logs into a success subdirectory
        local success_dir="${log_dir}/success"
        mkdir -p "$success_dir"
        local archived_stdout="$success_dir/$(basename "$stdout_log")"
        local archived_stderr="$success_dir/$(basename "$stderr_log")"
        mv "$stdout_log" "$archived_stdout"
        mv "$stderr_log" "$archived_stderr"

        # CSV (pipe-delimited) single-line entry
        if [ ! -s "$success_log" ]; then
            echo "timestamp|status|run_name|command|stdout_log|stderr_log" > "$success_log"
        fi
        local status_label="SUCCESS"
        $skipped && status_label="SKIPPED"
        printf "%s|%s|%s|%s|%s|%s\n" \
            "$timestamp" "$status_label" "$run_name" "$cmd" "$archived_stdout" "$archived_stderr" >> "$success_log"
        return 0
    else
        echo "❌ [FAILURE] Finished: $run_name"

        echo "----------------- ERROR OUTPUT -----------------"
        tail -n 15 "$stderr_log" # Now operates on the permanent log file
        echo "------------------------------------------------"

        # On failure, archive logs into a failure subdirectory
        local failed_dir="${log_dir}/failed"
        mkdir -p "$failed_dir"
        local failed_stdout="$failed_dir/$(basename "$stdout_log")"
        local failed_stderr="$failed_dir/$(basename "$stderr_log")"
        mv "$stdout_log" "$failed_stdout"
        mv "$stderr_log" "$failed_stderr"

        # CSV (pipe-delimited) single-line entry
        if [ ! -s "$failed_log" ]; then
            echo "timestamp|status|run_name|command|stdout_log|stderr_log" > "$failed_log"
        fi
        printf "%s|FAILURE|%s|%s|%s|%s\n" \
            "$timestamp" "$run_name" "$cmd" "$failed_stdout" "$failed_stderr" >> "$failed_log"
        return 1
    fi
}
