#!/bin/bash

# Anti-Stockfish: ULTIMATE Phased Learning System
# Optimized for: Apple M4 Pro, 24GB RAM, 14 CPU cores, Metal GPU

cd ~/anti-stockfish
source venv/bin/activate

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║         🏆 ANTI-STOCKFISH: ULTIMATE PHASED LEARNING SYSTEM 🏆               ║"
echo "║                                                                               ║"
echo "║  Optimized for: Apple M4 Pro (14 cores, 24GB RAM, Metal GPU)                ║"
echo "║                                                                               ║"
echo "║  Phase 1: Top 100 Super GMs (500 games) → Train                             ║"
echo "║  Phase 2: Historical Games → Retrain                                         ║"
echo "║  Phase 3: Top 1000 + Super GMs (500 games) → Retrain                        ║"
echo "║  Phase 4: Continuous Learning (+1000 games) → Keep Retraining               ║"
echo "║                                                                               ║"
echo "║  Goal: 100,000 games × 1,100 players = 110 MILLION GAMES!                   ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔥 Performance Features:"
echo "   ✅ Multi-threaded data collection (all 14 cores)"
echo "   ✅ Metal GPU acceleration for training"
echo "   ✅ Large batch sizes (256) with 24GB RAM"
echo "   ✅ Parallel game processing"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

# Run in background
nohup python3 ultimate_phased_learning.py > phased_learning.log 2>&1 &
echo $! > phased_learning.pid

echo ""
echo "✅ Ultimate phased learning started!"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f ~/anti-stockfish/phased_learning.log"
echo ""
echo "📊 Check status:"
echo "   ps -p \$(cat ~/anti-stockfish/phased_learning.pid)"
echo ""
echo "📊 Check state:"
echo "   cat ~/anti-stockfish/phased_learning_state.json | python3 -m json.tool"
echo ""
echo "⏹️  Stop:"
echo "   kill \$(cat ~/anti-stockfish/phased_learning.pid)"
echo ""
echo "🚀 LET'S BEAT STOCKFISH WITH M4 PRO POWER! 🚀"
