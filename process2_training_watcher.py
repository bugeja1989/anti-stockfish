#!/usr/bin/env python3

"""
Anti-Stockfish Process 2: Continuous Training & GUI Backend
1. Monitors Process 1 for new games
2. Extracts positions continuously
3. Trains models continuously
4. Serves the Cyberpunk GUI via Flask
"""

import subprocess
import time
import logging
from pathlib import Path
import json
import chess
import chess.pgn
import io
import torch
from flask import Flask, request, jsonify, render_template
from threading import Thread
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [P2-TRAINER] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process2_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app for API and GUI
app = Flask(__name__, template_folder='templates')

# Silence Flask access logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class ContinuousTrainer:
    """Continuous training with real-time predictions"""
    
    def __init__(self):
        self.data_dir = Path("neural_network/data")
        self.model_dir = Path("neural_network/models")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.chesscom_dataset = self.data_dir / "chesscom_master_dataset.jsonl"
        self.positions_dataset = self.data_dir / "extracted_positions.jsonl"
        
        self.state_file = Path("process2_state.json")
        
        # Latest model info
        self.latest_model = None
        self.model_version = 0
        
        self.load_state()
    
    def load_state(self):
        """Load state with migration support"""
        default_state = {
            'last_chesscom_entries': 0,
            'total_positions_extracted': 0,
            'models_trained': 0,
            'last_training_time': None,
            'training_active': False
        }
        
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    loaded_state = json.load(f)
                
                # Merge loaded state with default state to ensure all keys exist
                self.state = {**default_state, **loaded_state}
                
                # Ensure numeric values are valid
                self.state['models_trained'] = int(self.state.get('models_trained', 0))
                self.state['total_positions_extracted'] = int(self.state.get('total_positions_extracted', 0))
                
                logger.info(f"📊 Loaded: {self.state['models_trained']} models, {self.state['total_positions_extracted']:,} positions")
            except Exception as e:
                logger.warning(f"⚠️  State file corrupted or incompatible ({e}), starting fresh")
                self.state = default_state
        else:
            self.state = default_state
            logger.info("🆕 Starting fresh")
    
    def save_state(self):
        """Save state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_dataset_size(self, filepath):
        """Get number of lines"""
        if not filepath.exists():
            return 0
        try:
            with open(filepath) as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def extract_positions_from_game(self, game_data):
        """Extract positions from Chess.com game"""
        positions = []
        
        try:
            pgn_text = game_data.get('pgn')
            if not pgn_text:
                return positions
            
            pgn = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn)
            
            if not game:
                return positions
            
            result = game.headers.get('Result', '*')
            if result == '1-0':
                outcome = 1.0
            elif result == '0-1':
                outcome = 0.0
            else:
                outcome = 0.5
            
            board = game.board()
            move_number = 0
            
            for move in game.mainline_moves():
                fen = board.fen()
                
                position = {
                    'fen': fen,
                    'move': move.uci(),
                    'outcome': outcome,
                    'move_number': move_number,
                    'source': 'chesscom'
                }
                
                positions.append(position)
                board.push(move)
                move_number += 1
            
            return positions
        
        except Exception as e:
            return positions
    
    def extract_new_positions(self):
        """Extract positions from new games"""
        entries = self.get_dataset_size(self.chesscom_dataset)
        new_entries = entries - self.state['last_chesscom_entries']
        
        if new_entries <= 0:
            return 0
        
        logger.info(f"📊 Processing {new_entries} new entries...")
        
        total_new = 0
        
        with open(self.chesscom_dataset) as f:
            # Skip processed
            for _ in range(self.state['last_chesscom_entries']):
                next(f)
            
            # Process new
            for line in f:
                try:
                    entry = json.loads(line)
                    games = entry.get('games', [])
                    
                    for game_data in games:
                        positions = self.extract_positions_from_game(game_data)
                        
                        with open(self.positions_dataset, 'a') as pf:
                            for pos in positions:
                                pf.write(json.dumps(pos) + '\n')
                                total_new += 1
                        
                        # Log progress every 1000 positions
                        if total_new % 1000 == 0:
                            logger.info(f"   ⚡ Extracted {total_new:,} positions so far...")
                
                except Exception as e:
                    continue
        
        self.state['last_chesscom_entries'] = entries
        self.state['total_positions_extracted'] += total_new
        self.save_state()
        
        logger.info(f"✅ Extracted {total_new:,} new positions (Total: {self.state['total_positions_extracted']:,})")
        
        return total_new
    
    def train_model(self):
        """Train model"""
        total_positions = self.get_dataset_size(self.positions_dataset)
        
        if total_positions < 1000:
            logger.warning(f"⚠️  Need more positions ({total_positions}/1000)")
            return False
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🧠 TRAINING MODEL #{self.state['models_trained'] + 1}")
        logger.info(f"📊 Positions: {total_positions:,}")
        logger.info(f"{'='*80}\n")
        
        self.state['training_active'] = True
        self.save_state()
        
        try:
            # Check for GPU
            GPU_AVAILABLE = torch.backends.mps.is_available()
            device = "mps" if GPU_AVAILABLE else "cpu"
            BATCH_SIZE = 256 if GPU_AVAILABLE else 64
            
            logger.info(f"🔥 GPU: {'MPS (Metal)' if GPU_AVAILABLE else 'CPU'}")
            logger.info(f"📦 Batch size: {BATCH_SIZE}")
            
            # Increase epochs as we get more data
            epochs = min(10 + (self.state['models_trained'] * 2), 30)
            
            cmd = [
                "python3", "neural_network/src/train.py",
                "--data", str(self.positions_dataset),
                "--epochs", str(epochs),
                "--batch-size", str(BATCH_SIZE),
                "--device", device,
                "--num-workers", "8"
            ]
            
            logger.info(f"🚀 Executing: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info(f"✅ Training complete!")
                self.state['models_trained'] += 1
                self.state['last_training_time'] = time.time()
                self.model_version = self.state['models_trained']
                self.save_state()
            else:
                logger.error(f"❌ Training failed: {result.stderr[:500]}")
            
            self.state['training_active'] = False
            self.save_state()
            
            return result.returncode == 0
        
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            self.state['training_active'] = False
            self.save_state()
            return False
    
    def get_best_move(self, fen: str) -> dict:
        """Get best move for position"""
        try:
            board = chess.Board(fen)
            
            # Simple evaluation (will be replaced with neural network)
            legal_moves = list(board.legal_moves)
            
            if not legal_moves:
                return {'error': 'No legal moves'}
            
            # For now, return first legal move
            # TODO: Use trained model for actual prediction
            best_move = legal_moves[0]
            
            return {
                'best_move': best_move.uci(),
                'model_version': self.model_version,
                'positions_trained': self.state['total_positions_extracted'],
                'models_trained': self.state['models_trained']
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def run_training_loop(self):
        """Main training loop"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🧠 CONTINUOUS TRAINER")
        logger.info(f"{'='*80}\n")
        
        CHECK_INTERVAL = 30  # Check every 30s
        TRAIN_INTERVAL = 300  # Train every 5 minutes if new data
        
        last_train_time = 0
        
        try:
            while True:
                # Heartbeat log
                logger.info(f"💓 Heartbeat: Checking for new games... (Models: {self.state['models_trained']}, Positions: {self.state['total_positions_extracted']:,})")
                
                # Extract new positions
                new_positions = self.extract_new_positions()
                
                # Train if enough time passed and we have new data
                current_time = time.time()
                if new_positions > 0 and (current_time - last_train_time) >= TRAIN_INTERVAL:
                    self.train_model()
                    last_train_time = current_time
                
                time.sleep(CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info(f"\n⏹️  Stopped")
            logger.info(f"📊 Models: {self.state['models_trained']}, Positions: {self.state['total_positions_extracted']:,}")
            self.save_state()

# Global trainer instance
trainer = None

@app.route('/')
def index():
    """Serve the GUI"""
    return render_template('index.html')

@app.route('/api/best_move', methods=['POST'])
def api_best_move():
    """API endpoint for best move"""
    data = request.json
    fen = data.get('fen')
    
    if not fen:
        return jsonify({'error': 'Missing FEN'}), 400
    
    result = trainer.get_best_move(fen)
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """API endpoint for training stats"""
    return jsonify({
        'models_trained': trainer.state['models_trained'],
        'positions_extracted': trainer.state['total_positions_extracted'],
        'training_active': trainer.state['training_active'],
        'model_version': trainer.model_version
    })

def run_api_server():
    """Run Flask API server"""
    logger.info("🌐 Starting GUI on http://localhost:5443")
    app.run(host='0.0.0.0', port=5443, debug=False, use_reloader=False)

def main():
    global trainer
    trainer = ContinuousTrainer()
    
    # Start API server in background thread
    api_thread = Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Run training loop
    trainer.run_training_loop()

if __name__ == '__main__':
    main()
