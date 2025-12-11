#!/usr/bin/env python3

"""
Anti-Stockfish Process 2: Continuous Trainer (Backend Only)
1. Monitors Process 1 for new games
2. Extracts positions continuously
3. Trains models continuously
4. DOES NOT serve GUI (Separation of Concerns)
"""

import subprocess
import time
import logging
from pathlib import Path
import json
import chess
import chess.pgn
import io
import sys
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [TRAINER] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process2_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousTrainer:
    """Continuous training loop"""
    
    def __init__(self):
        self.data_dir = Path("neural_network/data")
        self.model_dir = Path("neural_network/models")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.chesscom_dataset = self.data_dir / "chesscom_master_dataset.jsonl"
        self.positions_dataset = self.data_dir / "extracted_positions.jsonl"
        
        self.state_file = Path("process2_state.json")
        
        self.load_state()
        
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
                self.state = {**default_state, **loaded_state}
                logger.info(f"📊 Loaded State: {self.state['models_trained']} models, {self.state['total_positions_extracted']:,} positions")
            except Exception as e:
                logger.warning(f"⚠️  State file corrupted ({e}), starting fresh")
                self.state = default_state
        else:
            self.state = default_state
            logger.info("🆕 Starting fresh")
    
    def save_state(self):
        """Save state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_dataset_size(self, filepath):
        if not filepath.exists(): return 0
        try:
            with open(filepath) as f: return sum(1 for _ in f)
        except: return 0
    
    def extract_positions_from_game(self, game_data):
        positions = []
        try:
            pgn_text = game_data.get('pgn')
            if not pgn_text: return positions
            
            pgn = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn)
            if not game: return positions
            
            board = game.board()
            result = game.headers.get("Result", "*")
            game_score = 1.0 if result == "1-0" else (0.0 if result == "0-1" else 0.5)
                
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
        if not self.chesscom_dataset.exists(): return 0
            
        current_lines = self.get_dataset_size(self.chesscom_dataset)
        new_lines = current_lines - self.state['last_chesscom_entries']
        
        if new_lines > 0:
            logger.info(f"📥 Found {new_lines} new games to process...")
            count = 0
            with open(self.chesscom_dataset) as f:
                for _ in range(self.state['last_chesscom_entries']): next(f)
                with open(self.positions_dataset, 'a') as out:
                    for line in f:
                        try:
                            game_data = json.loads(line)
                            positions = self.extract_positions_from_game(game_data)
                            for pos in positions:
                                out.write(json.dumps(pos) + '\n')
                                count += 1
                        except: continue
            
            self.state['last_chesscom_entries'] = current_lines
            self.state['total_positions_extracted'] += count
            self.save_state()
            logger.info(f"✅ Extracted {count} new positions. Total: {self.state['total_positions_extracted']:,}")
            return count
        return 0

    def run_training_cycle(self):
        if self.state['total_positions_extracted'] < 100: return False
            
        logger.info("🏋️  Starting Training Cycle...")
        self.state['training_active'] = True
        self.save_state()
        
        try:
            cmd = [sys.executable, "neural_network/src/train.py", "--epochs", "1", "--batch-size", "64"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.state['models_trained'] += 1
                self.state['last_training_time'] = time.time()
                self.state['current_version_tag'] = f"v{self.state['models_trained']}"
                self.save_state()
                logger.info(f"✅ Training Complete! New Version: {self.state['current_version_tag']}")
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

if __name__ == "__main__":
    trainer = ContinuousTrainer()
    logger.info("🚀 Trainer Service Started (Background)")
    
    while True:
        try:
            trainer.run_extraction_cycle()
            trainer.run_training_cycle()
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("🛑 Trainer Stopped by User")
            break
        except Exception as e:
            logger.error(f"Loop Error: {e}")
            time.sleep(5)
