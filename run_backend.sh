#!/bin/bash

echo "=================================================="
echo "🧠 Anti-Stockfish: BACKEND (Engine & Training)"
echo "=================================================="

# 1. Check Dependencies
if ! command -v stockfish &> /dev/null; then
    echo "❌ Stockfish not found. Please install it (brew install stockfish)."
    exit 1
fi

# 2. Install Python dependencies
echo "📦 Verifying Python dependencies..."
pip3 install -r requirements.txt > /dev/null 2>&1

# 3. Start Data Collector (Background)
echo "🕵️  Starting Data Collector (Process 1)..."
nohup python3 process1_chesscom_collector.py > process1.log 2>&1 &
PID1=$!
echo "   ✅ Collector running (PID: $PID1)"

# 4. Start Training & Inference Engine (Foreground)
echo "🏋️  Starting Training & Inference Engine (Process 2)..."
echo "   (Press Ctrl+C to stop)"
python3 process2_training_watcher.py

# Cleanup on exit
echo "🛑 Stopping Collector..."
kill $PID1
