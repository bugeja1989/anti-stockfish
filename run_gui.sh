#!/bin/bash

echo "=================================================="
echo "🖥️  Anti-Stockfish: FRONTEND (Web GUI)"
echo "=================================================="

# Start the GUI process (Inference Only)
echo "🌐 Starting Web Interface..."
echo "   (Open http://localhost:5443 in your browser)"
python3 process2_training_watcher.py --mode gui
