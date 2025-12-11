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
    echo "♟️  Anti-Stockfish: SERVICE MANAGER"
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
    echo "0. Stop ALL & Exit"
    echo "L. View Logs"
    echo "Q. Quit Menu (Keep services running)"
    echo "=================================================="
    read -p "Select an option: " choice

    case $choice in
        4) start_service "collector" "python3 process1_chesscom_collector.py" "process1.log" ;;
        5) stop_service "collector" ;;
        6) start_service "trainer" "python3 process2_training_watcher.py --mode training" "process2_training.log" ;;
        7) stop_service "trainer" ;;
        8) start_service "gui" "python3 process2_training_watcher.py --mode gui" "process2_gui.log" ;;
        9) stop_service "gui" ;;
        0) 
            stop_service "collector"
            stop_service "trainer"
            stop_service "gui"
            exit 0
            ;;
        L|l)
            echo "Which log? (1=Collector, 2=Trainer, 3=GUI)"
            read -p "> " log_choice
            case $log_choice in
                1) tail -f process1.log ;;
                2) tail -f process2_training.log ;;
                3) tail -f process2_gui.log ;;
            esac
            ;;
        Q|q) exit 0 ;;
        *) echo "Invalid option." ;;
    esac
    
    echo "Press Enter to continue..."
    read
done
