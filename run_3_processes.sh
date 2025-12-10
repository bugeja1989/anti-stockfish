#!/bin/bash

# Anti-Stockfish: 3-Process Continuous Learning System
# Process 1: Super GMs (Chess.com)
# Process 2: Top 1000 (Lichess)
# Process 3: Continuous Training Watcher

cd ~/anti-stockfish
source venv/bin/activate

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║         🚀 ANTI-STOCKFISH: 3-PROCESS SYSTEM 🚀                               ║"
echo "║                                                                               ║"
echo "║  Process 1: Super GMs from Chess.com (PREFERRED)                            ║"
echo "║  Process 2: Top 1000 from Lichess                                            ║"
echo "║  Process 3: Continuous Training Watcher                                      ║"
echo "║                                                                               ║"
echo "║  ✅ ONE player at a time (NO rate limiting!)                                 ║"
echo "║  ✅ Continuous training when new data arrives                                ║"
echo "║  ✅ Uses all cores + GPU for training                                        ║"
echo "║  ✅ Chess.com + Lichess = Best quality data!                                 ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

# Start Process 1: Super GMs (Chess.com)
echo "🏆 Starting Process 1: Super GMs Collector (Chess.com)..."
nohup python3 process1_super_gms.py > process1_super_gms.log 2>&1 &
echo $! > process1.pid
echo "   ✅ Process 1 started (PID: $(cat process1.pid))"

sleep 2

# Start Process 2: Top 1000 (Lichess)
echo "🌍 Starting Process 2: Top 1000 Collector (Lichess)..."
nohup python3 process2_top_1000.py > process2_top_1000.log 2>&1 &
echo $! > process2.pid
echo "   ✅ Process 2 started (PID: $(cat process2.pid))"

sleep 2

# Start Process 3: Training Watcher
echo "👁️  Starting Process 3: Training Watcher..."
nohup python3 process3_training_watcher.py > process3_training.log 2>&1 &
echo $! > process3.pid
echo "   ✅ Process 3 started (PID: $(cat process3.pid))"

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║                    ✅ ALL 3 PROCESSES RUNNING! ✅                            ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Monitor processes:"
echo "   Process 1: tail -f ~/anti-stockfish/process1_super_gms.log"
echo "   Process 2: tail -f ~/anti-stockfish/process2_top_1000.log"
echo "   Process 3: tail -f ~/anti-stockfish/process3_training.log"
echo ""
echo "📊 Check status:"
echo "   ps -p \$(cat ~/anti-stockfish/process1.pid) -p \$(cat ~/anti-stockfish/process2.pid) -p \$(cat ~/anti-stockfish/process3.pid)"
echo ""
echo "📊 Check states:"
echo "   cat ~/anti-stockfish/process1_state.json | python3 -m json.tool"
echo "   cat ~/anti-stockfish/process2_state.json | python3 -m json.tool"
echo "   cat ~/anti-stockfish/process3_state.json | python3 -m json.tool"
echo ""
echo "⏹️  Stop all:"
echo "   kill \$(cat ~/anti-stockfish/process1.pid) \$(cat ~/anti-stockfish/process2.pid) \$(cat ~/anti-stockfish/process3.pid)"
echo ""
echo "🚀 COLLECTING FROM CHESS.COM + LICHESS, TRAINING CONTINUOUSLY! 🚀"
