#!/bin/bash

echo "=================================================="
echo "♟️  Installing Stockfish for Anti-Stockfish Simulation"
echo "=================================================="

OS="$(uname -s)"

if [ "$OS" == "Darwin" ]; then
    echo "🍎 Detected macOS"
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
        exit 1
    fi
    
    echo "📦 Installing Stockfish via Homebrew..."
    brew install stockfish
    
    STOCKFISH_PATH=$(which stockfish)
    echo "✅ Stockfish installed at: $STOCKFISH_PATH"

elif [ "$OS" == "Linux" ]; then
    echo "🐧 Detected Linux"
    
    if command -v apt-get &> /dev/null; then
        echo "📦 Installing Stockfish via apt..."
        sudo apt-get update
        sudo apt-get install -y stockfish
    elif command -v yum &> /dev/null; then
        echo "📦 Installing Stockfish via yum..."
        sudo yum install -y stockfish
    else
        echo "⚠️  Unsupported package manager. Please install Stockfish manually."
        exit 1
    fi
    
    STOCKFISH_PATH=$(which stockfish)
    echo "✅ Stockfish installed at: $STOCKFISH_PATH"

else
    echo "⚠️  Unsupported OS: $OS"
    echo "Please install Stockfish manually from https://stockfishchess.org/download/"
    exit 1
fi

echo ""
echo "🎉 Installation Complete!"
echo "You can now run the simulation mode in the Anti-Stockfish GUI."
