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
import chess.engine
import io
import torch
from flask import Flask, request, jsonify, render_template
from threading import Thread
import sys
import re

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
        self.model_version = "v0"
        
        self.load_state()
        self.stockfish_path = self.find_stockfish()
        if self.stockfish_path:
            logger.info(f"✅ Stockfish found at: {self.stockfish_path}")
        else:
            logger.warning("⚠️  Stockfish NOT found! Simulation mode will fail.")
            
        # Load Opening Book
        self.opening_book = self.load_opening_book()

    def load_opening_book(self):
        """Load ECO opening book into memory"""
        book = {}
        files = ['ecoA.json', 'ecoB.json', 'ecoC.json', 'ecoD.json', 'ecoE.json']
        count = 0
        for filename in files:
            path = self.data_dir / filename
            if path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                        # Data is {FEN: {moves: "1. e4...", ...}}
                        # We need to map FEN -> Best Move (first move in 'moves' string relative to FEN?)
                        # Actually, the JSON keys are FENs. The 'moves' string is the full line.
                        # We need to parse the NEXT move from the FEN.
                        # However, the JSON structure is: FEN -> Opening Info.
                        # The FEN is the position *after* the moves? Or the starting position of the variation?
                        # Let's assume the FEN is the position we are IN.
                        # Wait, the JSON keys are FENs.
                        # If we are at FEN X, and it's in the book, what is the move?
                        # The 'moves' string in the JSON is the *history* that led to this FEN.
                        # It does NOT tell us the *next* move.
                        #
                        # BUT, we injected these into training data by playing through them.
                        # To use them as a lookup book, we need {FEN -> Next Move}.
                        # Since the JSON only gives us the FEN and the moves that got there,
                        # we can't easily know the *next* move unless we have a tree.
                        #
                        # ALTERNATIVE: We already injected these into `extracted_positions.jsonl`.
                        # But that's for training.
                        #
                        # BETTER APPROACH:
                        # We can build a simple FEN->Move map by iterating through the PGNs in the JSONs again.
                        # For each opening line "1. e4 e5 2. Nf3...", we play it.
                        # Position before 1. e4 -> Move e4
                        # Position after 1. e4 -> Move e5
                        # ...
                        # We store this in a dictionary.
                        
                        for op in data.values():
                            pgn_moves = op.get('moves', '')
                            if not pgn_moves: continue
                            
                            # We need to parse this PGN
                            # Since we don't want to re-parse everything on every startup (slow),
                            # maybe we should rely on the model being trained on this?
                            # The user said "I need the Openings to always be the models that is run for the first 10 moves".
                            # This implies strict lookup if possible.
                            # Parsing 12k games on startup might take 10-20 seconds. Acceptable.
                            pass 
                            
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
        
        # Actually, let's implement the parsing logic properly in a separate method or just do it here.
        # To avoid blocking startup too long, we can do it in a background thread or just accept the delay.
        # Let's do it here for simplicity.
        
        logger.info("📚 Building Opening Book Lookup Table (this may take a moment)...")
        lookup_table = {}
        
        for filename in files:
            path = self.data_dir / filename
            if path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                        for op in data.values():
                            pgn_moves = op.get('moves', '')
                            if not pgn_moves: continue
                            
                            try:
                                game = chess.pgn.read_game(io.StringIO(pgn_moves))
                                if not game: continue
                                
                                board = game.board()
                                for move in game.mainline_moves():
                                    # Normalize FEN: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                                    # We want: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
                                    # Split by space and take first 4 parts
                                    fen = board.fen()
                                    fen_norm = " ".join(fen.split(" ")[:4])
                                    
                                    # Store the move for this FEN
                                    lookup_table[fen_norm] = move.uci()
                                    board.push(move)
                                    
                                    # Stop after 15 moves to keep table size manageable?
                                    # User said "first 10 moves". Let's go up to 15.
                                    if board.fullmove_number > 15:
                                        break
                            except:
                                continue
                except:
                    pass
        
        logger.info(f"📚 Opening Book Ready: {len(lookup_table):,} positions loaded.")
        return lookup_table

    def find_stockfish(self):
        """Locate stockfish executable"""
        import shutil
        path = shutil.which("stockfish")
        if path:
            return path
        # Common paths
        paths = [
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "/opt/homebrew/bin/stockfish"
        ]
        for p in paths:
            if Path(p).exists():
                return p
        return None
    
    def load_state(self):
        """Load state with migration support"""
        default_state = {
            'last_chesscom_entries': 0,
            'total_positions_extracted': 0,
            'models_trained': 0,
            'last_training_time': None,
            'training_active': False,
            'current_version_tag': "v0"
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
                self.model_version = self.state.get('current_version_tag', "v0")
                
                logger.info(f"📊 Loaded: {self.state['models_trained']} models, {self.state['total_positions_extracted']:,} positions, Version: {self.model_version}")
            except Exception as e:
                logger.warning(f"⚠️  State file corrupted or incompatible ({e}), starting fresh")
                self.state = default_state
        else:
            self.state = default_state
            logger.info("🆕 Starting fresh")
    
    def save_state(self):
        """Save state"""
        self.state['current_version_tag'] = self.model_version
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
            move_number = 1  # Start from move 1
            
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
        
        # Limit batch size to allow training to happen more often
        BATCH_LIMIT = 5000  # Process max 5000 games per cycle to catch up with collector
        entries_to_process = min(new_entries, BATCH_LIMIT)
        
        logger.info(f"📊 Processing {entries_to_process} new entries (Batch limit: {BATCH_LIMIT})...")
        
        total_new = 0
        processed_count = 0
        
        with open(self.chesscom_dataset) as f:
            # Skip processed
            for _ in range(self.state['last_chesscom_entries']):
                next(f)
            
            # Process new (up to limit)
            for line in f:
                if processed_count >= BATCH_LIMIT:
                    break
                
                processed_count += 1
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
                            # Update state in real-time so GUI sees progress
                            self.state['total_positions_extracted'] += 1000
                            self.save_state()
                
                except Exception as e:
                    continue
        
        self.state['last_chesscom_entries'] += processed_count
        # Sync total positions with actual file size
        real_total = self.get_dataset_size(self.positions_dataset)
        self.state['total_positions_extracted'] = real_total
        self.save_state()
        
        logger.info(f"✅ Extracted {total_new:,} new positions (Total: {real_total:,})")
        
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
            # Use larger batch size for M4 Pro to maximize throughput
            BATCH_SIZE = 1024 if GPU_AVAILABLE else 64
            
            logger.info(f"🔥 GPU: {'MPS (Metal)' if GPU_AVAILABLE else 'CPU'}")
            logger.info(f"📦 Batch size: {BATCH_SIZE}")
            
            # Fixed epochs as requested
            epochs = 6
            
            cmd = [
                "python3", "neural_network/src/train.py",
                "--data", str(self.positions_dataset),
                "--epochs", str(epochs),
                "--batch-size", str(BATCH_SIZE),
                "--device", device,
                "--num-workers", "8",
                "--resume",
                "--run-id", str(self.state['models_trained'] + 1)
            ]
            
            logger.info(f"🚀 Executing: {' '.join(cmd)}")
            
            # Stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Regex to capture version tag from logs
            # Example: "Epoch 1/10 (Version v1.1)"
            version_pattern = re.compile(r"\(Version v(\d+\.\d+)\)")
            
            # Read output line by line
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"   🔥 {line}")
                    
                    # Check for version update
                    match = version_pattern.search(line)
                    if match:
                        # Capture only the number part (e.g., "1.1")
                        new_version = "v" + match.group(1)
                        if new_version != self.model_version:
                            self.model_version = new_version
                            self.save_state()
                            logger.info(f"   🏷️  GUI Version Updated: {self.model_version}")
            
            process.wait()
            
            if process.returncode == 0:
                logger.info(f"✅ Training complete!")
                self.state['models_trained'] += 1
                self.state['last_training_time'] = time.time()
                # Ensure final version is saved
                self.save_state()
            else:
                logger.error(f"❌ Training failed with code {process.returncode}")
            
            self.state['training_active'] = False
            self.save_state()
            
            return process.returncode == 0
        
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            self.state['training_active'] = False
            self.save_state()
            return False
    
    def get_stockfish_move(self, fen: str, elo: int) -> dict:
        """Get move from Stockfish at specific ELO"""
        if not self.stockfish_path:
            return {'error': 'Stockfish not found'}
            
        try:
            board = chess.Board(fen)
            
            # Limit ELO (Stockfish supports UCI_LimitStrength and UCI_Elo)
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                # Configure ELO
                if elo < 500:
                    # For very low ELO, we also limit nodes to ensure it plays badly
                    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
                    result = engine.play(board, chess.engine.Limit(nodes=100)) # Limit nodes for stupidity
                else:
                    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
                    # Time limit based on ELO (give it a bit more time for higher ELOs)
                    time_limit = 0.1 if elo < 1500 else 0.5 if elo < 2500 else 1.0
                    result = engine.play(board, chess.engine.Limit(time=time_limit))
                
                return {'best_move': result.move.uci(), 'elo': elo}
                
        except Exception as e:
            logger.error(f"Stockfish error: {e}")
            return {'error': str(e)}

    def get_best_move(self, fen: str, pgn: str = None) -> dict:
        """Get best move for position using the latest trained model"""
        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            
            # Parse PGN to detect repetitions
            history_fens = []
            if pgn:
                try:
                    pgn_io = io.StringIO(pgn)
                    game = chess.pgn.read_game(pgn_io)
                    if game:
                        temp_board = game.board()
                        history_fens.append(temp_board.fen().split(' ')[0]) # Store position only (no clocks)
                        for move in game.mainline_moves():
                            temp_board.push(move)
                            history_fens.append(temp_board.fen().split(' ')[0])
                        # Ensure we are at the current position
                        board = temp_board
                except Exception as e:
                    logger.warning(f"Failed to parse PGN for repetition check: {e}")
            
            if not legal_moves:
                return {'error': 'No legal moves'}
            
            # ---------------------------------------------------------
            # 1. OPENING BOOK LOOKUP (Strict Mode for Moves 1-10)
            # ---------------------------------------------------------
            # Check book BEFORE loading model to save time and ensure reliability
            move_count = board.fullmove_number
            if move_count <= 15 and hasattr(self, 'opening_book'): # Extended to 15 moves
                # Normalize current FEN for lookup
                fen_norm = " ".join(fen.split(" ")[:4])
                book_move_uci = self.opening_book.get(fen_norm)
                
                if book_move_uci:
                    # Verify legality just in case
                    if chess.Move.from_uci(book_move_uci) in board.legal_moves:
                        logger.info(f"📖 Book Move Found: {book_move_uci} (Move {move_count})")
                        return {
                            'best_move': book_move_uci,
                            'eval': 0.6, # Book value
                            'model_version': "Opening Book (ECO)",
                            'positions_trained': self.state['total_positions_extracted'],
                            'models_trained': self.state['models_trained']
                        }

            # ---------------------------------------------------------
            # 2. MODEL LOADING
            # ---------------------------------------------------------
            # Load model if not loaded or if version changed
            model_path = self.model_dir / f"chaos_module_{self.model_version}.pth"
            latest_path = self.model_dir / "chaos_module_latest.pth"
            
            # Check if we need to load/reload the model
            should_reload = False
            if not self.latest_model:
                should_reload = True
            elif hasattr(self, 'loaded_version') and self.loaded_version != self.model_version:
                should_reload = True
                logger.info(f"🔄 New version detected ({self.model_version}). Reloading model...")
            
            if should_reload:
                target_path = model_path if model_path.exists() else latest_path
                
                if target_path.exists():
                    try:
                        # Import model class dynamically to avoid circular imports
                        sys.path.append('neural_network/src')
                        from model import ChaosModule
                        
                        device = "mps" if torch.backends.mps.is_available() else "cpu"
                        self.latest_model = ChaosModule().to(device)
                        self.latest_model.load_state_dict(torch.load(target_path, map_location=device))
                        self.latest_model.eval()
                        self.loaded_version = self.model_version  # Track loaded version
                        logger.info(f"🧠 Loaded model: {target_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to load model: {e}")
                        self.latest_model = None
                else:
                    logger.warning(f"⚠️ Model file not found: {target_path}")

            # If no model available yet, fallback to random legal move
            if not self.latest_model:
                import random
                return {
                    'best_move': random.choice(legal_moves).uci(),
                    'eval': 0.0,
                    'model_version': self.model_version + " (Random Fallback)",
                    'positions_trained': self.state['total_positions_extracted'],
                    'models_trained': self.state['models_trained']
                }
            
            # ---------------------------------------------------------
            # 2. NEURAL NETWORK INFERENCE
            # ---------------------------------------------------------
            
            # Prepare input for model
            # We need to evaluate ALL legal moves and pick the best one
            best_move = None
            best_eval = -float('inf')
            
            # Helper function to convert board to tensor (copied from model.py to avoid import issues)
            def board_to_tensor(fen):
                import numpy as np
                board = chess.Board(fen)
                tensor = np.zeros((12, 8, 8), dtype=np.float32)
                piece_idx = {
                    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
                }
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece:
                        row = 7 - (square // 8)
                        col = square % 8
                        channel = piece_idx[piece.piece_type]
                        if piece.color == chess.BLACK:
                            channel += 6
                        tensor[channel, row, col] = 1.0
                return torch.from_numpy(tensor)

            # Evaluate all legal moves (1-ply search)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            
            for move in legal_moves:
                board.push(move)
                fen_after = board.fen()
                fen_simple = fen_after.split(' ')[0] # Position only
                
                # Check for repetition
                repetition_count = history_fens.count(fen_simple)
                
                # Convert to tensor
                tensor = board_to_tensor(fen_after).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    policy, value, chaos = self.latest_model(tensor)
                    
                    # Score = Value (positional) + Chaos (complexity bonus)
                    # We invert value if it's Black's turn (since model predicts White's advantage)
                    score = value.item()
                    if board.turn == chess.BLACK:
                        score = -score
                        
                    # Add chaos bonus (Anti-Stockfish prefers chaos!)
                    # STRICT OPENING MODE: If we are out of book but still in first 10 moves,
                    # we use ZERO chaos to ensure we play solid chess (Value Head only).
                    # After move 10, we unleash the chaos.
                    
                    if move_count <= 10:
                        chaos_weight = 0.0  # Pure solid play in opening
                    else:
                        chaos_weight = 0.1  # Chaos mode activated
                        
                    score += (chaos.item() * chaos_weight)
                    
                    # REPETITION PENALTY
                    # If position has occurred 2+ times, apply massive penalty to avoid 3-fold repetition
                    if repetition_count >= 2:
                        score -= 1000.0 # NUCLEAR penalty (Draw is forbidden)
                        logger.info(f"🚫 Avoiding 3-fold repetition: {move.uci()} (Count: {repetition_count})")
                    elif repetition_count == 1:
                        score -= 5.0 # Stronger penalty for 2nd occurrence to prevent shuffling
                        # If we are in chaos mode, we might tolerate it slightly, but generally we want progress.
                        # But for Anti-Stockfish, shuffling is death.
                        logger.info(f"⚠️ Discouraging repetition: {move.uci()}")
                
                if score > best_eval:
                    best_eval = score
                    best_move = move
                
                board.pop()
            
            return {
                'best_move': best_move.uci(),
                'eval': round(best_eval, 4),
                'model_version': self.model_version,
                'positions_trained': self.state['total_positions_extracted'],
                'models_trained': self.state['models_trained']
            }
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {'error': str(e)}

# Global trainer instance
trainer = ContinuousTrainer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    return jsonify({
        'models_trained': trainer.state['models_trained'],
        'positions_extracted': trainer.state['total_positions_extracted'],
        'model_version': trainer.model_version,
        'gpu_status': 'METAL ACTIVE' if torch.backends.mps.is_available() else 'CPU ONLY',
        'training_active': trainer.state['training_active']
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    fen = data.get('fen')
    pgn = data.get('pgn') # Get PGN for repetition check
    
    if not fen:
        return jsonify({'error': 'No FEN provided'}), 400
    
    result = trainer.get_best_move(fen, pgn)
    return jsonify(result)

@app.route('/api/stockfish', methods=['POST'])
def stockfish_move():
    data = request.json
    fen = data.get('fen')
    elo = int(data.get('elo', 1500))
    
    if not fen:
        return jsonify({'error': 'No FEN provided'}), 400
        
    result = trainer.get_stockfish_move(fen, elo)
    return jsonify(result)

def run_flask():
    app.run(host='0.0.0.0', port=5443, debug=False, use_reloader=False)

def main_loop():
    """Main loop for Process 2"""
    logger.info("🧠 CONTINUOUS TRAINER")
    logger.info("="*80)
    
    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("🌐 Starting GUI on http://localhost:5443")
    
    while True:
        try:
            logger.info(f"💓 Heartbeat: Checking for new games... (Models: {trainer.state['models_trained']}, Positions: {trainer.state['total_positions_extracted']:,})")
            
            # 1. Extract new positions
            new_positions = trainer.extract_new_positions()
            
            # 2. Train if we have enough data
            if new_positions > 0 or trainer.state['models_trained'] == 0:
                trainer.train_model()
            
            # Sleep before next check
            time.sleep(10)
            
        except KeyboardInterrupt:
            logger.info("🛑 Stopping Process 2...")
            break
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")
            time.sleep(10)

if __name__ == '__main__':
    main_loop()
