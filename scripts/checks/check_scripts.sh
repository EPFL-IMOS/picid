#!/bin/bash

# Define log files
LOG_FILE="scripts/checks/script_output.log"
ERROR_LOG_FILE="scripts/checks/script_error.log"
FAILED_LOG_FILE="scripts/checks/failed_runs.log"

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

MODELS=("lstm" "cnn_1d" "crossformer" "stf" "patchtst" "tide" "timeseries_transformer" "carte_fit_predict" "tabpfn_fit_predict" "tabdpt_fit_predict" "xgboost_fit_predict")

for model in "${MODELS[@]}"; do
    echo "Testing model: $model"
    
    # Create a list of commands to execute for each model
    commands=(
        # "python lmetk/run.py experiment=unibo/prognostics/bosello23_base_statistics/${model}.yaml trainer.max_epochs=1"
        "python lmetk/run.py experiment=nb14/prognostics/statistics/${model}.yaml trainer.max_epochs=1"
        # "python lmetk/run.py experiment=pronostia/prognostics/statistics/${model}.yaml trainer.max_epochs=1"
        # "python lmetk/run.py experiment=xjtu_sy/prognostics/statistics/${model}.yaml trainer.max_epochs=1"
    )
    
    # Iterate over the commands and run them
    for cmd in "${commands[@]}"; do
        echo "Running: $cmd"
        run_command "$cmd"
    done

done
