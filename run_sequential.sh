#!/bin/bash

# Anti-Stockfish: SEQUENTIAL One-by-One Learning
# ONE player → TRAIN → repeat

cd ~/anti-stockfish
source venv/bin/activate

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║            🎯 ANTI-STOCKFISH: SEQUENTIAL LEARNING 🎯                         ║"
echo "║                                                                               ║"
echo "║  Strategy: ONE player at a time, TRAIN after EACH                           ║"
echo "║                                                                               ║"
echo "║  1. Collect from Magnus Carlsen (DrNykterstein)                             ║"
echo "║  2. Train model → Model learns from Magnus                                   ║"
echo "║  3. Collect from Hikaru Nakamura                                             ║"
echo "║  4. Train model → Model learns from Hikaru (now smarter!)                    ║"
echo "║  5. Repeat for all 100 Super GMs                                             ║"
echo "║                                                                               ║"
echo "║  Result: Model gets smarter after EVERY player!                              ║"
echo "║                                                                               ║"
echo "║  ✅ NO rate limiting (one at a time)                                         ║"
echo "║  ✅ Continuous improvement                                                    ║"
echo "║  ✅ Uses all cores + GPU for training                                        ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

# Run in background
nohup python3 sequential_learning.py > sequential_learning.log 2>&1 &
echo $! > sequential_learning.pid

echo ""
echo "✅ Sequential learning started!"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f ~/anti-stockfish/sequential_learning.log"
echo ""
echo "📊 Check status:"
echo "   ps -p \$(cat ~/anti-stockfish/sequential_learning.pid)"
echo ""
echo "📊 Check state:"
echo "   cat ~/anti-stockfish/sequential_learning_state.json | python3 -m json.tool"
echo ""
echo "⏹️  Stop:"
echo "   kill \$(cat ~/anti-stockfish/sequential_learning.pid)"
echo ""
echo "🚀 ONE PLAYER AT A TIME, GETTING SMARTER EVERY STEP! 🚀"
