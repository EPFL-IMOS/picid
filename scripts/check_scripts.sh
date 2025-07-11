#!/bin/bash

# Define log files
LOG_FILE="scripts/script_output.log"
ERROR_LOG_FILE="scripts/script_error.log"
FAILED_LOG_FILE="scripts/failed_runs.log"

# Clear previous log files
> $LOG_FILE
> $ERROR_LOG_FILE
> $FAILED_LOG_FILE

# Function to run a command and check for failure
run_command() {
    local cmd="$1"
    
    # Run the command and capture the output and error
    { eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; } 2>> "$ERROR_LOG_FILE"
    
    # Check if the command failed
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Command failed: $cmd" >> "$FAILED_LOG_FILE"
        echo "Check $ERROR_LOG_FILE for details." >> "$FAILED_LOG_FILE"
    fi
}

# List of commands to execute
commands=(
    "python lmetk/run.py experiment=battery/nb1_stf trainer.max_epochs=5"
    "python lmetk/run.py experiment=battery/nb1_tabpfn"
    "python lmetk/run.py experiment=battery/nb14_stf trainer.max_epochs=5"
    "python lmetk/run.py experiment=battery/nb14_tabpfn"
    "python lmetk/run.py experiment=railway_traction/timepfn"
    "python lmetk/run.py experiment=railway_traction/tabpfn_railway_seasonal_context"
    "python lmetk/run.py experiment=railway_traction/tabpfn_no_target_context"
    "python lmetk/run.py experiment=concepts_n_cmapss/tabpfn"
 
)

# Iterate over the commands and run them
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    run_command "$cmd"
done






