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
# Construct list of PIDs to check, removing empty ones
PIDS_TO_CHECK=""
[ -n "$P1_PID" ] && PIDS_TO_CHECK="$P1_PID"
[ -n "$P2_PID" ] && PIDS_TO_CHECK="${PIDS_TO_CHECK:+$PIDS_TO_CHECK,}$P2_PID"
[ -n "$TRAIN_PID" ] && PIDS_TO_CHECK="${PIDS_TO_CHECK:+$PIDS_TO_CHECK,}$TRAIN_PID"

if [ -n "$PIDS_TO_CHECK" ]; then
    ps -p "$PIDS_TO_CHECK" -o pid,pcpu,pmem,rss,command | awk '
    BEGIN {printf "%-8s %-8s %-8s %-10s %s\n", "PID", "CPU%", "MEM%", "RSS(MB)", "COMMAND"}
    NR>1 {printf "%-8s %-8s %-8s %-10.2f %s\n", $1, $2, $3, $4/1024, $5}'
else
    echo "⚠️  No active Anti-Stockfish processes found to analyze."
fi

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
    
    # Run powermetrics once and save to temp file
    # Using -i 1000 (1s) and -n 1 to get a quick snapshot, but wait 5s to let it stabilize if needed?
    # Actually, powermetrics needs a sample interval. Let's do 1 sample over 1000ms.
    powermetrics --samplers cpu_power,gpu_power,thermal -n 1 -i 1000 > /tmp/as_metrics.txt 2>/dev/null
    
    echo "--- POWER & FREQUENCY ---"
    grep -E "CPU Power|GPU Power|ANE Power|Package Power" /tmp/as_metrics.txt || echo "Power metrics not found."
    
    echo ""
    echo "--- GPU UTILIZATION ---"
    # Try case-insensitive grep for "active residency" or "GPU active"
    grep -iE "GPU active|active residency" /tmp/as_metrics.txt || echo "GPU utilization not found."

    echo ""
    echo "--- THERMALS ---"
    # M-series often outputs "Fan: 0 RPM" or just "Fan"
    # Also look for "Average die temperature"
    grep -iE "Fan|die temperature|Headroom" /tmp/as_metrics.txt | grep -v "sensor" | head -n 10
    
    echo ""
    echo "--- ANE (NEURAL ENGINE) STATUS ---"
    ANE_POWER=$(grep "ANE Power" /tmp/as_metrics.txt)
    if [ -z "$ANE_POWER" ]; then
        echo "ANE Power: 0 mW (Idle or not reported)"
    else
        echo "$ANE_POWER"
        echo "ℹ️  Note: PyTorch MPS uses the GPU. ANE is mostly for CoreML."
    fi
fi

echo ""
echo "================================================================================"
