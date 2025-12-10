#!/bin/bash

echo "================================================================================"
echo "📊 ANTI-STOCKFISH SYSTEM MONITOR (macOS)"
echo "================================================================================"

# Check if processes are running
echo ""
echo "🔍 PROCESS STATUS:"
P1_PID=$(pgrep -f "process1_chesscom_collector.py")
P2_PID=$(pgrep -f "process2_training_watcher.py")
TRAIN_PID=$(pgrep -f "neural_network/src/train.py")

if [ -z "$P1_PID" ]; then
    echo "❌ Process 1 (Collector): NOT RUNNING"
else
    echo "✅ Process 1 (Collector): RUNNING (PID: $P1_PID)"
fi

if [ -z "$P2_PID" ]; then
    echo "❌ Process 2 (Trainer):   NOT RUNNING"
else
    echo "✅ Process 2 (Trainer):   RUNNING (PID: $P2_PID)"
fi

if [ -z "$TRAIN_PID" ]; then
    echo "💤 Training Job:          IDLE (Waiting for next batch)"
else
    echo "🔥 Training Job:          ACTIVE (PID: $TRAIN_PID)"
fi

# CPU & Memory Usage for Python processes
echo ""
echo "💻 PYTHON RESOURCE USAGE:"
ps -eo pid,pcpu,pmem,comm | grep "python" | grep -v grep | awk '{print "PID: "$1" | CPU: "$2"% | MEM: "$3"% | "$4}'

# GPU Usage (requires sudo for powermetrics, falling back to top if not root)
echo ""
echo "🚀 GPU & SYSTEM LOAD:"
if [ "$EUID" -ne 0 ]; then
    echo "(Run with 'sudo ./check_status.sh' to see detailed GPU power usage)"
    top -l 1 | grep -E "^CPU|^PhysMem"
else
    echo "Gathering GPU stats (sampling 1s)..."
    powermetrics --samplers gpu_power -n 1 -i 1000 | grep -E "GPU Power|GPU Active|GPU Idle"
fi

echo ""
echo "================================================================================"
