#!/bin/bash

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

MAX_PARALLEL=2
job_counter=0
pids=()

run_and_log() {
    local cmd="$1"
    local name="$2"

    echo "Launching: $cmd  (log: $LOG_DIR/$name.log)"
    eval "$cmd" > "$LOG_DIR/$name.log" 2>&1 &
    pid=$!
    pids+=($pid)
}

# Example: launch 5 dummy jobs that take different times
for i in {1..5}; do
    run_and_log "sleep $((3+i)) && echo Job $i done" "job_$i"
    ((job_counter++))

    if [[ "$job_counter" -ge "$MAX_PARALLEL" ]]; then
        # Wait until any one finishes
        wait -n 
        ((job_counter--))
    fi
done


# Wait for remaining jobs
wait
echo "All jobs finished."