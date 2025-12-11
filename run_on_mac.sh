#!/bin/bash

echo "=================================================="
echo "🚀 Anti-Stockfish: FULL SYSTEM LAUNCH"
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

# 4. Start Continuous Trainer (Background)
echo "🏋️  Starting Continuous Trainer (Process 2)..."
nohup python3 process2_trainer.py > process2_training.log 2>&1 &
PID2=$!

# 5. Start Web GUI (Foreground)
echo "🌐 Starting Web Interface (Port 5443)..."
echo "   (Open http://localhost:5443 in your browser)"
echo "   (Press Ctrl+C to stop all services)"

trap "kill $PID1 $PID2; exit" INT
python3 web_gui.py
