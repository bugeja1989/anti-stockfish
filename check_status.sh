#!/bin/bash

echo "================================================================================"
echo "🔬 ANTI-STOCKFISH DEEP DIAGNOSTICS (macOS M-Series)"
echo "================================================================================"

# 1. PROCESS HEALTH
echo ""
echo "🔍 PROCESS STATUS:"
P1_PID=$(pgrep -f "process1_chesscom_collector.py")
P2_PID=$(pgrep -f "process2_training_watcher.py")
TRAIN_PID=$(pgrep -f "neural_network/src/train.py")

if [ -z "$P1_PID" ]; then echo "❌ Process 1 (Collector): NOT RUNNING"; else echo "✅ Process 1 (Collector): RUNNING (PID: $P1_PID)"; fi
if [ -z "$P2_PID" ]; then echo "❌ Process 2 (Trainer):   NOT RUNNING"; else echo "✅ Process 2 (Trainer):   RUNNING (PID: $P2_PID)"; fi
if [ -z "$TRAIN_PID" ]; then echo "💤 Training Job:          IDLE"; else echo "🔥 Training Job:          ACTIVE (PID: $TRAIN_PID)"; fi

# 2. MEMORY PRESSURE
echo ""
echo "🧠 MEMORY & SWAP:"
vm_stat | grep "Pages free"
vm_stat | grep "Pages active"
sysctl vm.swapusage

# 3. PYTHON RESOURCE USAGE
echo ""
echo "💻 PYTHON PROCESS DETAILS:"
ps -eo pid,pcpu,pmem,rss,comm | grep "python" | grep -v grep | awk '
BEGIN {printf "%-8s %-8s %-8s %-10s %s\n", "PID", "CPU%", "MEM%", "RSS(MB)", "COMMAND"}
{printf "%-8s %-8s %-8s %-10.2f %s\n", $1, $2, $3, $4/1024, $5}'

# 4. DEEP HARDWARE STATS (Requires Sudo)
echo ""
echo "🚀 HARDWARE TELEMETRY (CPU/GPU/ANE/THERMALS):"
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  NOTE: Run with 'sudo' to see Power (Watts), Frequency, and Thermals!"
    echo "   Example: sudo ./check_status.sh"
    echo ""
    echo "--- Standard Stats ---"
    top -l 1 | grep -E "^CPU|^PhysMem"
else
    echo "Gathering 5 seconds of telemetry..."
    # Capture CPU, GPU, ANE, and Thermal stats
    powermetrics --samplers cpu_power,gpu_power,thermal -n 1 -i 5000 --format plist > /tmp/anti_stockfish_metrics.xml
    
    # Parse key metrics (using grep for simplicity as plist parsing in bash is verbose)
    echo "--- POWER & FREQUENCY ---"
    powermetrics --samplers cpu_power,gpu_power -n 1 -i 1000 | grep -E "CPU Power|GPU Power|Package Power|ANE Power|GPU Active|GPU Idle"
    
    echo ""
    echo "--- THERMALS ---"
    powermetrics --samplers thermal -n 1 -i 1000 | grep -E "Fan|Temperature"
fi

echo ""
echo "================================================================================"
