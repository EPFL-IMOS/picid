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

MODELS=("lstm" "stf" "patchtst")  #"crossformer" "tide" "timeseries_transformer" "carte_fit_predict" "tabpfn_fit_predict" "tabdpt_fit_predict" "xgboost_fit_predict")
# "lstm"  "stf" "patchtst" "cnn_1d"  "stf" "patchtst"


# Create a list of commands to execute for each model

# 1. Use SINGLE QUOTES (') to define the commands.
# This prevents ${model} from being expanded immediately.
commands=(

    'python lmetk/run.py experiment=pronostia/prognostics/raw/${model}.yaml task_definition.seq_len=2560 trainer.max_epochs=1000  datamodule.val_batch_size=1024 datamodule.train_batch_size=512 optimization.lr=0.001 optimization.optimizer.weight_decay=1e-5 evaluator.val.log_image=False' # model.n_layers=2 model.hidden_dim=128'
    # 'python lmetk/run.py experiment=pronostia/prognostics/spectral/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.01 task_definition.seq_len=16' # model.n_layers=2 model.hidden_dim=128 debug=pronostia
    # 'python lmetk/run.py experiment=pronostia/prognostics/spectral/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.001 task_definition.seq_len=16' # model.n_layers=2 model.hidden_dim=128 debug=pronostia

    #'python lmetk/run.py experiment=unibo/prognostics/bosello23_base_statistics/${model}.yaml trainer.max_epochs=1'
    #'python lmetk/run.py experiment=nb14/prognostics/statistics/${model}.yaml trainer.max_epochs=1'
    #'python lmetk/run.py experiment=pronostia/prognostics/statistics/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.01 task_definition.seq_len=16'
    #'python lmetk/run.py experiment=pronostia/prognostics/statistics/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.001 task_definition.seq_len=16'
    # 'python lmetk/run.py experiment=pronostia/prognostics/statistics/${model}.yaml trainer.max_epochs=250 optimizer.parameters.lr=0.01'
    # 'python lmetk/run.py experiment=pronostia/prognostics/statistics/${model}.yaml trainer.max_epochs=250 optimizer.parameters.lr=0.01 optimizer.parameters.weight_decay=0.0001'
    #'python lmetk/run.py experiment=xjtu_sy/prognostics/statistics/${model}.yaml trainer.max_epochs=25 debug=pronostia'
)

# 2. Keep the loop order the same (commands outer, models inner).
for cmd_template in "${commands[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Testing model: $model"
        
        # 3. Use 'eval' to substitute the $model variable into the command template.
        # This creates the final command string to be executed.
        eval "final_cmd=\"$cmd_template\""
        
        # 4. Run the fully-formed command.
        run_command "$final_cmd"
    done
done


