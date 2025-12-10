#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║         🛑 ANTI-STOCKFISH: EMERGENCY STOP 🛑                                 ║"
echo "║                                                                               ║"
echo "║  Killing all processes...                                                     ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Kill by PID files if they exist
if [ -f process1.pid ]; then
    PID=$(cat process1.pid)
    echo "🔪 Killing Process 1 (PID: $PID)..."
    kill -9 $PID 2>/dev/null
    rm process1.pid
fi

if [ -f process2.pid ]; then
    PID=$(cat process2.pid)
    echo "🔪 Killing Process 2 (PID: $PID)..."
    kill -9 $PID 2>/dev/null
    rm process2.pid
fi

# 2. Kill by process name (Main Scripts)
echo "🔪 Hunting down stray Python scripts..."
pkill -9 -f "process1_chesscom_collector.py"
pkill -9 -f "process2_training_watcher.py"
pkill -9 -f "neural_network/src/train.py"

# 3. Kill multiprocessing workers (Aggressive)
echo "🔪 Cleaning up worker threads..."
pkill -9 -f "multiprocessing.spawn"
pkill -9 -f "multiprocessing.resource_tracker"

# 4. Free port 5443
echo "🔓 Freeing port 5443..."
lsof -ti:5443 | xargs kill -9 2>/dev/null

echo ""
echo "✅ ALL SYSTEMS STOPPED."
