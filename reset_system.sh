#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║         ⚠️  ANTI-STOCKFISH: FACTORY RESET ⚠️                                 ║"
echo "║                                                                               ║"
echo "║  This will DELETE ALL DATA, MODELS, and LOGS.                                 ║"
echo "║  Are you sure? (Waiting 5 seconds...)                                         ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

sleep 5

echo "🧹 Stopping all processes..."
./run_2_processes.sh stop_only 2>/dev/null
# Manual kill just in case
pkill -9 -f "process1_chesscom_collector.py"
pkill -9 -f "process2_training_watcher.py"
pkill -9 -f "neural_network/src/train.py"
pkill -9 -f "multiprocessing.spawn"

echo "🗑️  Deleting Data..."
rm -rf neural_network/data/*
rm -rf neural_network/models/*

echo "🗑️  Deleting State Files..."
rm -f process1_state.json
rm -f process2_state.json
rm -f process1.pid
rm -f process2.pid

echo "🗑️  Deleting Logs..."
rm -f process1_chesscom.log
rm -f process2_training.log

echo "✨ System Reset Complete! You can now run ./run_2_processes.sh to start fresh."
