#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║         🏆 ANTI-STOCKFISH: 2-PROCESS SYSTEM 🏆                               ║"
echo "║                                                                               ║"
echo "║  Optimized for: Apple M4 Pro (14 cores, 24GB RAM, Metal GPU)                ║"
echo "║                                                                               ║"
echo "║  Process 1: Chess.com Collector (Target: 100M+ Positions)                   ║"
echo "║  Process 2: Continuous Trainer + Cyberpunk GUI (localhost:5443)             ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Function to check and install dependencies
check_dependency() {
    PACKAGE=$1
    if ! python3 -c "import $PACKAGE" &> /dev/null; then
        echo "⚠️  Missing dependency: $PACKAGE"
        echo "📦 Installing $PACKAGE..."
        pip3 install $PACKAGE
        echo "✅ Installed $PACKAGE"
    else
        echo "✅ Found $PACKAGE"
    fi
}

# CLEANUP: Aggressively kill existing processes
echo "🧹 Cleaning up previous runs..."

# 1. Kill by PID files if they exist
if [ -f process1.pid ]; then
    kill -9 $(cat process1.pid) 2>/dev/null
    rm process1.pid
fi
if [ -f process2.pid ]; then
    kill -9 $(cat process2.pid) 2>/dev/null
    rm process2.pid
fi

# 2. Kill by process name (Main Scripts)
pkill -9 -f "process1_chesscom_collector.py"
pkill -9 -f "process2_training_watcher.py"
pkill -9 -f "neural_network/src/train.py"

# 3. Kill multiprocessing workers (Aggressive)
# This finds any python process that is a child of the above or related to multiprocessing
# We look for "multiprocessing.spawn" or "resource_tracker" which are common in PyTorch DataLoader
pkill -9 -f "multiprocessing.spawn"
pkill -9 -f "multiprocessing.resource_tracker"

# 4. Free port 5443 (macOS/Linux)
echo "🔓 Freeing port 5443..."
lsof -ti:5443 | xargs kill -9 2>/dev/null

# Wait for cleanup to actually happen
sleep 3
echo "✅ Cleanup complete! All old processes should be dead."
echo ""

echo "🔍 Checking dependencies..."
check_dependency "flask"
check_dependency "chess"
check_dependency "torch"
check_dependency "requests"
check_dependency "numpy"
echo "✅ All dependencies ready!"
echo ""

# Start Process 1: Chess.com Collector
echo "🚀 Starting Process 1: Chess.com Collector..."
nohup python3 process1_chesscom_collector.py > process1_chesscom.log 2>&1 &
echo $! > process1.pid
echo "✅ Process 1 started (PID: $(cat process1.pid))"
echo ""

# Start Process 2: Training Watcher + GUI
echo "🚀 Starting Process 2: Trainer + GUI..."
nohup python3 process2_training_watcher.py > process2_training.log 2>&1 &
echo $! > process2.pid
echo "✅ Process 2 started (PID: $(cat process2.pid))"
echo ""

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║  ✅ ALL PROCESSES STARTED!                                                   ║"
echo "║                                                                               ║"
echo "║  👉 OPEN GUI: http://localhost:5443                                          ║"
echo "║                                                                               ║"
echo "║  Monitor logs:                                                               ║"
84	echo "║    tail -f process1_chesscom.log                                             ║"
85	echo "║    tail -f process2_training.log                                             ║"
86	echo "║                                                                               ║"
87	echo "║  Stop all:                                                                   ║"
88	echo "║    kill \$(cat process1.pid) \$(cat process2.pid)                                ║"
89	echo "║                                                                               ║"
90	echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
