#!/bin/bash
# run_and_monitor.sh
# Usage: ./runai.sh [args...]

ARGS="$@"

if [[ -z "$ARGS" ]]; then
  echo "Usage: $0 experiment=..."
  exit 1
fi

# Prepare environment
source /workspace/setup/.venv/bin/activate
cd /workspace/projects/PICID || exit 1
uv pip install -e ".[dev]"

mkdir -p logs
LOGFILE="logs/run_experiment_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename $LOGFILE)" logs/latest.log

echo -e "\033[1;34m[INFO]\033[0m Running experiment with args: $ARGS"
echo -e "\033[1;34m[INFO]\033[0m Logs: $LOGFILE"

# Launch experiment
python -u picid/run.py paths=runai $ARGS 2>&1 | tee "$LOGFILE" &
PID=$!

# Also kill child process on Ctrl+C
trap "kill $PID 2>/dev/null" INT

# Monitor loop With signal 0, no signal is actually delivered.
# The call can be used to check for the existence of a process ID.
while kill -0 $PID 2>/dev/null; do
  clear
  echo "==== Monitoring experiment (PID $PID) ===="
  tail -n 20 "$LOGFILE"
  echo "------------------------------------------"
  ps -o %cpu,%mem,etime -p $PID | tail -n 1
  echo "Press Ctrl+C to stop monitoring"
  sleep 2
done

wait $PID
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
  echo -e "\033[1;32m[INFO]\033[0m Process finished successfully."
else
  echo -e "\033[1;31m[ERROR]\033[0m Process exited with code $EXIT_CODE."
fi

echo "Full log in $LOGFILE"
