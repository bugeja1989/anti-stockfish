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

# CLEANUP: Kill existing processes and free port 5000
echo "🧹 Cleaning up previous runs..."

# Kill by PID files if they exist
if [ -f process1.pid ]; then
    kill $(cat process1.pid) 2>/dev/null
    rm process1.pid
fi
if [ -f process2.pid ]; then
    kill $(cat process2.pid) 2>/dev/null
    rm process2.pid
fi

# Kill by process name (just in case)
pkill -f "process1_chesscom_collector.py"
pkill -f "process2_training_watcher.py"

# Free port 5443 (macOS/Linux)
echo "🔓 Freeing port 5443..."
lsof -ti:5443 | xargs kill -9 2>/dev/null

# Wait for cleanup
sleep 2
echo "✅ Cleanup complete!"
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
echo "║    tail -f process1_chesscom.log                                             ║"
echo "║    tail -f process2_training.log                                             ║"
echo "║                                                                               ║"
echo "║  Stop all:                                                                   ║"
echo "║    kill \$(cat process1.pid) \$(cat process2.pid)                                ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
