#!/bin/bash

echo "=================================================="
echo "♟️  Anti-Stockfish: Mac Setup & Launch Script"
echo "=================================================="

# 1. Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
    exit 1
fi

# 2. Install Stockfish if missing
if ! command -v stockfish &> /dev/null; then
    echo "📦 Installing Stockfish via Homebrew..."
    brew install stockfish
else
    echo "✅ Stockfish is already installed."
fi

# 3. Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# 4. Inject Opening Book (The 12,000+ positions!)
echo "📚 Injecting 12,000+ Opening Book positions..."
python3 inject_opening_book.py

# 5. Start the Engine
echo "🚀 Starting Anti-Stockfish Engine..."
echo "   - Process 1: Game Collector (Background)"
echo "   - Process 2: Training & GUI (Foreground)"

# Start Process 1 in background
nohup python3 process1_chesscom_collector.py > process1.log 2>&1 &
PID1=$!
echo "   ✅ Collector started (PID: $PID1)"

# Start Process 2 in foreground
python3 process2_training_watcher.py

# Cleanup on exit
kill $PID1
