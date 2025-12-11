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
                    if board.fullmove_number > 10:
                        score += (chaos.item() * 0.1)
                    
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
                    
                    if down_material and chaos.item() > 0.5:
                        # "Trust the Chaos" - recover some of the lost material score
                        # We add a "Compensation Bonus"
                        compensation = 0.5 * chaos.item() # Up to 0.5 pawn worth of "faith"
                        if is_white:
                            score += compensation
                        else:
                            score -= compensation
                            
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
            # Usually engines display score relative to side to move.
            display_eval = best_eval if maximizing_player else -best_eval

            return {
                'best_move': best_move.uci(),
                'eval': round(display_eval, 4),
                'model_version': self.model_version,
                'positions_trained': self.state['total_positions_extracted'],
                'models_trained': self.state['models_trained'],
                'depth': SEARCH_DEPTH
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
    # Reload state from disk to get latest updates from the Training process
    trainer.load_state()
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
    parser = argparse.ArgumentParser(description='Anti-Stockfish Engine & Trainer')
    parser.add_argument('--mode', type=str, default='both', choices=['training', 'gui', 'both'],
                        help='Mode to run: "training" (backend only), "gui" (frontend only), or "both" (default)')
    args = parser.parse_args()

    logger.info(f"🧠 ANTI-STOCKFISH (Mode: {args.mode.upper()})")
    logger.info("="*80)
    
    # GUI MODE or BOTH
    if args.mode in ['gui', 'both']:
        # Start Flask
        # If 'gui' mode, we run Flask in the main thread (blocking)
        # If 'both' mode, we run Flask in a separate thread
        if args.mode == 'gui':
            logger.info("🌐 Starting GUI on http://localhost:5443 (Inference Only)")
            run_flask() # Blocking
            return # Exit when Flask stops
        else:
            flask_thread = Thread(target=run_flask, daemon=True)
            flask_thread.start()
            logger.info("🌐 Starting GUI on http://localhost:5443")

    # TRAINING MODE or BOTH
    if args.mode in ['training', 'both']:
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
