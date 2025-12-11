#!/bin/bash

echo "=================================================="
echo "♟️  Anti-Stockfish: MASTER LAUNCHER"
echo "=================================================="

# 1. Check Dependencies
if ! command -v stockfish &> /dev/null; then
    echo "❌ Stockfish not found. Please install it (brew install stockfish)."
    exit 1
fi

# 2. Install Python dependencies
pip3 install -r requirements.txt > /dev/null 2>&1

# 3. Inject Opening Book (if needed)
if [ ! -f "neural_network/data/extracted_positions.jsonl" ]; then
    echo "📚 Injecting Opening Book..."
    python3 inject_opening_book.py
fi

# 4. Start Processes
echo "🚀 Starting System Modules..."

# Process 1: Collector (Background)
nohup python3 process1_chesscom_collector.py > process1.log 2>&1 &
PID1=$!
echo "   ✅ [1/3] Data Collector started (PID: $PID1)"

# Process 2: Trainer (Background)
nohup python3 process2_training_watcher.py --mode training > process2_training.log 2>&1 &
PID2=$!
echo "   ✅ [2/3] AI Trainer started (PID: $PID2)"

# Process 3: GUI (Foreground)
echo "   ✅ [3/3] Web GUI starting..."
echo "   (Press Ctrl+C to stop everything)"
echo "=================================================="
python3 process2_training_watcher.py --mode gui

# Cleanup on exit
echo "🛑 Stopping background processes..."
kill $PID1 $PID2
