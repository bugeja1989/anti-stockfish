#!/bin/bash

PID_DIR=".pids"
mkdir -p $PID_DIR

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

get_pid() {
    if [ -f "$PID_DIR/$1.pid" ]; then
        cat "$PID_DIR/$1.pid"
    else
        echo ""
    fi
}

check_status() {
    pid=$(get_pid $1)
    if [ -n "$pid" ] && ps -p $pid > /dev/null; then
        echo -e "${GREEN}RUNNING (PID: $pid)${NC}"
        return 0
    else
        echo -e "${RED}STOPPED${NC}"
        return 1
    fi
}

start_service() {
    service=$1
    cmd=$2
    log=$3
    
    if check_status $service > /dev/null; then
        echo "⚠️  $service is already running."
    else
        echo "🚀 Starting $service..."
        nohup $cmd > $log 2>&1 &
        echo $! > "$PID_DIR/$service.pid"
        echo "✅ Started."
    fi
}

stop_service() {
    service=$1
    pid=$(get_pid $service)
    
    if [ -n "$pid" ]; then
        echo "🛑 Stopping $service (PID: $pid)..."
        kill $pid 2>/dev/null
        rm "$PID_DIR/$service.pid"
        echo "✅ Stopped."
    else
        echo "⚠️  $service is not running."
    fi
}

while true; do
    clear
    echo "=================================================="
    echo "♟️  Anti-Stockfish: SERVICE MANAGER (v2.0)"
    echo "=================================================="
    echo -e "1. Collector (Data Mining)   [$(check_status collector)]"
    echo -e "2. Trainer   (AI Learning)   [$(check_status trainer)]"
    echo -e "3. GUI       (Web Interface) [$(check_status gui)]"
    echo "--------------------------------------------------"
    echo "4. Start Collector"
    echo "5. Stop Collector"
    echo "--------------------------------------------------"
    echo "6. Start Trainer"
    echo "7. Stop Trainer"
    echo "--------------------------------------------------"
    echo "8. Start GUI"
    echo "9. Stop GUI"
    echo "--------------------------------------------------"
    echo "A. Start ALL"
    echo "S. Stop ALL"
    echo "--------------------------------------------------"
    echo "H. Health Check & Fix"
    echo "R. Factory Reset (Wipe Data)"
    echo "L. View Logs"
    echo "Q. Quit Menu"
    echo "=================================================="
    read -p "Select an option: " choice

    case $choice in
        R|r)
            echo -e "${RED}⚠️  DANGER ZONE: FACTORY RESET ⚠️${NC}"
            echo "This will DELETE ALL training data, models, and logs."
            read -p "Are you absolutely sure? (type 'yes' to confirm): " confirm
            
            if [[ $confirm == "yes" ]]; then
                echo "🛑 Stopping all services..."
                stop_service "collector"
                stop_service "trainer"
                stop_service "gui"
                
                echo "🗑️  Deleting data..."
                rm -f neural_network/data/extracted_positions.jsonl
                rm -f neural_network/models/*.pth
                rm -f process2_state.json
                rm -f *.log
                
                echo "✅ System Reset Complete."
                echo "💡 Tip: Run 'H' (Health Check) now to re-inject the Opening Book!"
            else
                echo "❌ Reset cancelled."
            fi
            echo "Press Enter to continue..."
            read
            ;;
        H|h)
            echo "🏥 Running System Health Check..."
            
            # 1. Check Stockfish
            if ! command -v stockfish &> /dev/null; then
                echo "❌ Stockfish missing."
                read -p "Install via Homebrew? (y/n) " ans
                if [[ $ans == "y" ]]; then brew install stockfish; fi
            else
                echo "✅ Stockfish found."
            fi
            
            # 2. Check Python Deps
            echo "📦 Checking Python dependencies..."
            pip3 install -r requirements.txt > /dev/null 2>&1
            if [ $? -eq 0 ]; then echo "✅ Dependencies OK."; else echo "❌ Dependency install failed."; fi
            
            # 3. Check Opening Book
            if [ ! -f "neural_network/data/extracted_positions.jsonl" ]; then
                echo "❌ Opening Book missing."
                read -p "Inject Opening Book (12k positions)? (y/n) " ans
                if [[ $ans == "y" ]]; then python3 inject_opening_book.py; fi
            else
                lines=$(wc -l < "neural_network/data/extracted_positions.jsonl")
                echo "✅ Opening Book present ($lines positions)."
            fi
            
            # 4. Check Tal Games
            echo "❓ Have you injected the Tal/Kasparov games?"
            read -p "Inject them now? (y/n) " ans
            if [[ $ans == "y" ]]; then python3 inject_attacking_games.py; fi
            
            echo "✅ Health Check Complete. Press Enter."
            read
            ;;
        4) start_service "collector" "python3 process1_chesscom_collector.py" "process1.log" ;;
        5) stop_service "collector" ;;
        6) start_service "trainer" "python3 process2_trainer.py" "process2_training.log" ;;
        7) stop_service "trainer" ;;
        8) start_service "gui" "python3 web_gui.py" "web_gui.log" ;;
        9) stop_service "gui" ;;
        A|a)
            start_service "collector" "python3 process1_chesscom_collector.py" "process1.log"
            start_service "trainer" "python3 process2_trainer.py" "process2_training.log"
            start_service "gui" "python3 web_gui.py" "web_gui.log"
            ;;
        S|s)
            stop_service "collector"
            stop_service "trainer"
            stop_service "gui"
            ;;
        L|l)
            echo "Which log? (1=Collector, 2=Trainer, 3=GUI)"
            read -p "> " log_choice
            case $log_choice in
                1) tail -f process1.log ;;
                2) tail -f process2_training.log ;;
                3) tail -f web_gui.log ;;
            esac
            ;;
        Q|q) exit 0 ;;
        *) echo "Invalid option." ;;
    esac
    
    echo "Press Enter to continue..."
    read
done
