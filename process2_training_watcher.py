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
import argparse

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
            
            board = game.board()
            
            # Get result score
            result = game.headers.get("Result", "*")
            if result == "1-0":
                game_score = 1.0
            elif result == "0-1":
                game_score = 0.0
            else:
                game_score = 0.5
                
            # Extract FENs
            for move in game.mainline_moves():
                board.push(move)
                positions.append({
                    'fen': board.fen(),
                    'score': game_score,
                    'source': 'chesscom_import'
                })
                
        except Exception as e:
            logger.error(f"Error extracting positions: {e}")
            
        return positions

    def run_extraction_cycle(self):
        """Check for new games and extract positions"""
        if not self.chesscom_dataset.exists():
            return 0
            
        current_lines = self.get_dataset_size(self.chesscom_dataset)
        new_lines = current_lines - self.state['last_chesscom_entries']
        
        if new_lines > 0:
            logger.info(f"📥 Found {new_lines} new games to process...")
            
            count = 0
            with open(self.chesscom_dataset) as f:
                # Skip processed lines
                for _ in range(self.state['last_chesscom_entries']):
                    next(f)
                
                # Process new lines
                with open(self.positions_dataset, 'a') as out:
                    for line in f:
                        try:
                            game_data = json.loads(line)
                            positions = self.extract_positions_from_game(game_data)
                            for pos in positions:
                                out.write(json.dumps(pos) + '\n')
                                count += 1
                        except:
                            continue
            
            self.state['last_chesscom_entries'] = current_lines
            self.state['total_positions_extracted'] += count
            self.save_state()
            logger.info(f"✅ Extracted {count} new positions. Total: {self.state['total_positions_extracted']:,}")
            return count
        
        return 0

    def run_training_cycle(self):
        """Run one epoch of training if we have enough data"""
        if self.state['total_positions_extracted'] < 100:
            return False
            
        logger.info("🏋️  Starting Training Cycle...")
        self.state['training_active'] = True
        self.save_state()
        
        try:
            # Call the training script as a subprocess
            # We use the separate train.py script to keep memory clean
            cmd = [sys.executable, "neural_network/src/train.py", "--epochs", "1", "--batch-size", "64"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.state['models_trained'] += 1
                self.state['last_training_time'] = time.time()
                
                # Update version tag
                self.model_version = f"v{self.state['models_trained']}"
                self.save_state()
                
                logger.info(f"✅ Training Complete! New Version: {self.model_version}")
                logger.info(f"Output: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"❌ Training Failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Training Error: {e}")
            return False
        finally:
            self.state['training_active'] = False
            self.save_state()

    def get_best_move(self, fen, history_fens=[]):
        """Get best move using Opening Book -> Neural Net -> Stockfish Fallback"""
        
        # ---------------------------------------------------------
        # 1. OPENING BOOK LOOKUP (Moves 1-10)
        # ---------------------------------------------------------
        board = chess.Board(fen)
        move_count = board.fullmove_number
        
        # Normalize FEN for lookup (remove move clocks)
        fen_norm = " ".join(fen.split(" ")[:4])
        
        if move_count <= 10:
            if fen_norm in self.opening_book:
                book_move_uci = self.opening_book[fen_norm]
                legal_moves = list(board.legal_moves)
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
        legal_moves = list(board.legal_moves)
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
        # 2. SMART SEARCH (Alpha-Beta Pruning + Policy Guidance)
        # ---------------------------------------------------------
        
        # Helper function to convert board to tensor
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

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Evaluation Function (Leaf Node)
        def evaluate_position(board):
            fen = board.fen()
            tensor = board_to_tensor(fen).unsqueeze(0).to(device)
            with torch.no_grad():
                policy, value, chaos = self.latest_model(tensor)
                
                # Base Score: Value Head
                score = value.item()
                
                # Chaos Bonus (Anti-Stockfish Personality)
                # Only apply chaos if NOT in opening (Moves > 10)
                chaos_bonus = 0.0
                if board.fullmove_number > 10:
                    chaos_bonus = (chaos.item() * 0.1)
                    score += chaos_bonus
                
                # ---------------------------------------------------------
                # 3. MATERIAL SACRIFICE HEURISTIC (The "Tal" Logic)
                # ---------------------------------------------------------
                # If we are down material but have high chaos/activity, boost score.
                # This encourages the engine to "believe" in its sacrifices.
                
                # Simple material count
                piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
                white_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
                black_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())
                
                mat_diff = white_mat - black_mat
                
                # If we are White and down material (mat_diff < 0)
                # OR if we are Black and down material (mat_diff > 0)
                # AND chaos score is high (> 0.5), we assume it's a brilliant sacrifice.
                
                is_white = (board.turn == chess.WHITE)
                down_material = (mat_diff < -1 if is_white else mat_diff > 1)
                
                sac_bonus = 0.0
                if down_material and chaos.item() > 0.5:
                    # "Trust the Chaos" - recover some of the lost material score
                    # We add a "Compensation Bonus"
                    # TUNED DOWN: Was 0.5, now 0.2 to prevent reckless sacrifices
                    compensation = 0.2 * chaos.item() 
                    if is_white:
                        sac_bonus = compensation
                        score += compensation
                    else:
                        sac_bonus = -compensation
                        score -= compensation
                
                # LOGGING FOR DEBUGGING
                # Only log for the root move search to avoid spam, but we are inside recursion here.
                # We can't easily log every leaf.
                # But we can attach these metrics to the returned score if we changed the return type,
                # but Minimax expects a float.
                # So we just return the score.
                        
                # Perspective: Always return score from WHITE's perspective for Minimax
                return score

        # Alpha-Beta Search
        def alpha_beta(board, depth, alpha, beta, maximizing_player):
            if depth == 0 or board.is_game_over():
                return evaluate_position(board)
            
            legal_moves = list(board.legal_moves)
            
            # Move Ordering: Captures and Checks first (simple heuristic)
            # Ideally we use Policy Head here, but for speed we use simple heuristics first
            legal_moves.sort(key=lambda m: board.is_capture(m) or board.is_check(), reverse=True)
            
            if maximizing_player:
                max_eval = -float('inf')
                for move in legal_moves:
                    board.push(move)
                    eval = alpha_beta(board, depth - 1, alpha, beta, False)
                    board.pop()
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = float('inf')
                for move in legal_moves:
                    board.push(move)
                    eval = alpha_beta(board, depth - 1, alpha, beta, True)
                    board.pop()
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval

        # Root Search
        best_move = None
        best_eval = -float('inf')
        
        # Depth 3 Search (Fast but deeper than 1-ply)
        SEARCH_DEPTH = 3
        alpha = -float('inf')
        beta = float('inf')
        
        # Root Move Ordering: Use Policy Head to sort candidate moves!
        # This is CRITICAL for finding the best move quickly
        root_tensor = board_to_tensor(board.fen()).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, _, _ = self.latest_model(root_tensor)
            # We can't easily map logits to moves without the move map from training.
            # For now, we fallback to capture-heuristic ordering at root too.
            pass

        legal_moves.sort(key=lambda m: board.is_capture(m) or board.is_check(), reverse=True)
        
        maximizing_player = (board.turn == chess.WHITE)
        
        # Detailed Logging for Root Moves
        logger.info(f"🔍 Searching Best Move for {board.fen()} (Depth {SEARCH_DEPTH})")
        
        for move in legal_moves:
            board.push(move)
            
            # Check for repetition immediately at root
            fen_simple = board.fen().split(' ')[0]
            repetition_count = history_fens.count(fen_simple)
            
            if repetition_count >= 2:
                # Nuclear penalty for 3-fold
                eval = -1000.0 if maximizing_player else 1000.0
            elif repetition_count == 1:
                # Strong penalty for 2-fold
                penalty = 5.0
                if maximizing_player:
                    eval = alpha_beta(board, SEARCH_DEPTH - 1, alpha, beta, False) - penalty
                else:
                    eval = alpha_beta(board, SEARCH_DEPTH - 1, alpha, beta, True) + penalty
            else:
                # Normal search
                if maximizing_player:
                    eval = alpha_beta(board, SEARCH_DEPTH - 1, alpha, beta, False)
                else:
                    eval = alpha_beta(board, SEARCH_DEPTH - 1, alpha, beta, True)
            
            # Log the evaluation for this candidate move
            logger.info(f"   Move {move.uci()}: Eval={eval:.4f}")

            board.pop()
            
            # Update Best Move
            # If we are White, we want Max Eval. If Black, we want Min Eval.
            # But `best_eval` variable tracks the score for the *current turn player*.
            # So we always want to maximize `eval` relative to our perspective?
            # No, standard Minimax:
            # If White (Max): Pick move with highest eval.
            # If Black (Min): Pick move with lowest eval.
            
            if maximizing_player:
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
            else:
                # For Black, "best" means lowest score (most negative)
                # But we need to store it. Let's initialize best_eval differently for Black.
                if best_move is None or eval < best_eval:
                    best_eval = eval
                    best_move = move
                beta = min(beta, eval)
        
        # If Black, best_eval is negative. For display, we might want to flip it?
        # The GUI expects +1.0 for White winning, -1.0 for Black winning.
        # Our model outputs 0.0 to 1.0 (Sigmoid).
        # Wait, our model output is 0-1 (Win Prob).
        # But our AlphaBeta treats it as a score.
        # If model says 0.8, that's good for White.
        # If model says 0.2, that's good for Black.
        # So Maximize for White, Minimize for Black is correct.
        
        if best_move:
            logger.info(f"🏆 Best Move: {best_move.uci()} (Eval: {best_eval:.4f})")
            return {
                'best_move': best_move.uci(),
                'eval': best_eval,
                'model_version': self.model_version,
                'positions_trained': self.state['total_positions_extracted'],
                'models_trained': self.state['models_trained']
            }
        else:
            return {
                'best_move': None,
                'eval': 0.0,
                'model_version': self.model_version,
                'positions_trained': self.state['total_positions_extracted'],
                'models_trained': self.state['models_trained']
            }

# Global Trainer Instance
trainer = ContinuousTrainer()

# Background Thread for Continuous Operations
def background_loop():
    while True:
        try:
            # 1. Extract new positions
            trainer.run_extraction_cycle()
            
            # 2. Train if enough data
            trainer.run_training_cycle()
            
            # Sleep to avoid CPU hogging
            time.sleep(5)
        except Exception as e:
            logger.error(f"Background Loop Error: {e}")
            time.sleep(5)

# Start Background Thread
t = Thread(target=background_loop, daemon=True)
t.start()

# API Endpoints
@app.route('/api/stats', methods=['GET'])
def get_stats():
    # Reload state from disk to get latest numbers from background thread
    trainer.load_state()
    return jsonify(trainer.state)

@app.route('/api/best_move', methods=['POST'])
def best_move():
    data = request.json
    fen = data.get('fen')
    history = data.get('history', []) # List of FENs for repetition check
    
    if not fen:
        return jsonify({'error': 'No FEN provided'}), 400
        
    result = trainer.get_best_move(fen, history)
    return jsonify(result)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
