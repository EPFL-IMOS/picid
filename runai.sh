#!/bin/bash
# run_and_monitor.sh
# Usage: ./run_and_monitor.sh myscript.py [args...]

SCRIPT=$1
shift
ARGS="$@"

LOGFILE="run_$(basename $SCRIPT)_$(date +%Y%m%d_%H%M%S).log"

if [[ -z "$SCRIPT" ]]; then
  echo "Usage: $0 script.py [args...]"
  exit 1
fi

echo "[INFO] Running $SCRIPT with args: $ARGS"
echo "[INFO] Logs: $LOGFILE"

# Run python script in background, capture output
python -u "$SCRIPT" $ARGS 2>&1 | tee "$LOGFILE" &
PID=$!

# Simple monitor loop
while kill -0 $PID 2>/dev/null; do
  clear
  echo "==== Monitoring $SCRIPT (PID $PID) ===="
  tail -n 20 "$LOGFILE"
  echo "---------------------------------------"
  echo "Press Ctrl+C to stop monitoring"
  sleep 2
done

echo "[INFO] Process finished. Full log in $LOGFILE"