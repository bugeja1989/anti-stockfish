#!/usr/bin/env python3

"""
Anti-Stockfish Process 3: Continuous Training Watcher
Monitors data from Process 1 & 2, trains when new data arrives
"""

import subprocess
import time
import logging
from pathlib import Path
import json
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [P3-TRAINER] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process3_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingWatcher:
    """Watches for new data and trains continuously"""
    
    def __init__(self):
        self.data_dir = Path("neural_network/data")
        self.model_dir = Path("neural_network/models")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.super_gms_dataset = self.data_dir / "super_gms_dataset.jsonl"
        self.top_1000_dataset = self.data_dir / "top_1000_dataset.jsonl"
        self.master_dataset = self.data_dir / "master_dataset.jsonl"
        
        self.state_file = Path("process3_state.json")
        
        # Hardware
        self.GPU_AVAILABLE = torch.backends.mps.is_available()
        self.BATCH_SIZE = 256 if self.GPU_AVAILABLE else 64
        
        logger.info(f"🖥️  GPU: {self.GPU_AVAILABLE}, Batch: {self.BATCH_SIZE}")
        
        self.load_state()
    
    def load_state(self):
        """Load state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
            logger.info(f"📊 Loaded: {self.state['models_trained']} models trained")
        else:
            self.state = {
                'last_super_gms_size': 0,
                'last_top_1000_size': 0,
                'last_master_size': 0,
                'models_trained': 0,
                'last_training_time': None
            }
            logger.info("🆕 Starting fresh")
    
    def save_state(self):
        """Save state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_dataset_size(self, filepath):
        """Get number of lines in dataset"""
        if not filepath.exists():
            return 0
        try:
            with open(filepath) as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def merge_datasets(self):
        """Merge all datasets into master"""
        logger.info("📊 Merging datasets...")
        
        total_lines = 0
        
        with open(self.master_dataset, 'w') as master:
            # Add Super GMs data
            if self.super_gms_dataset.exists():
                with open(self.super_gms_dataset) as f:
                    for line in f:
                        master.write(line)
                        total_lines += 1
            
            # Add Top 1000 data
            if self.top_1000_dataset.exists():
                with open(self.top_1000_dataset) as f:
                    for line in f:
                        master.write(line)
                        total_lines += 1
        
        logger.info(f"✅ Merged: {total_lines:,} total positions")
        return total_lines
    
    def check_for_new_data(self):
        """Check if new data has arrived"""
        super_gms_size = self.get_dataset_size(self.super_gms_dataset)
        top_1000_size = self.get_dataset_size(self.top_1000_dataset)
        
        super_gms_new = super_gms_size > self.state['last_super_gms_size']
        top_1000_new = top_1000_size > self.state['last_top_1000_size']
        
        if super_gms_new or top_1000_new:
            logger.info(f"📥 New data detected!")
            logger.info(f"   Super GMs: {self.state['last_super_gms_size']:,} → {super_gms_size:,}")
            logger.info(f"   Top 1000: {self.state['last_top_1000_size']:,} → {top_1000_size:,}")
            
            self.state['last_super_gms_size'] = super_gms_size
            self.state['last_top_1000_size'] = top_1000_size
            
            return True
        
        return False
    
    def train_model(self):
        """Train model with all available data"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🧠 TRAINING MODEL #{self.state['models_trained'] + 1}")
        logger.info(f"{'='*80}\n")
        
        # Merge datasets
        total_positions = self.merge_datasets()
        
        if total_positions < 1000:
            logger.warning(f"⚠️  Not enough data yet ({total_positions} positions)")
            return False
        
        logger.info(f"📊 Total positions: {total_positions:,}")
        logger.info(f"🔥 GPU: {'MPS (Metal)' if self.GPU_AVAILABLE else 'CPU'}")
        logger.info(f"📦 Batch size: {self.BATCH_SIZE}\n")
        
        try:
            device = "mps" if self.GPU_AVAILABLE else "cpu"
            
            # Increase epochs as we get more data
            epochs = min(10 + (self.state['models_trained'] * 2), 30)
            
            cmd = [
                "python3", "neural_network/src/train.py",
                "--data", str(self.master_dataset),
                "--epochs", str(epochs),
                "--batch-size", str(self.BATCH_SIZE),
                "--device", device,
                "--num-workers", "8"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info(f"✅ Training complete!")
                self.state['models_trained'] += 1
                self.state['last_training_time'] = time.time()
                self.state['last_master_size'] = total_positions
                self.save_state()
                return True
            else:
                logger.error(f"❌ Training failed: {result.stderr[:200]}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            return False
    
    def run(self):
        """Main loop: Watch for data, train when available"""
        logger.info(f"\n{'='*80}")
        logger.info(f"👁️  PROCESS 3: CONTINUOUS TRAINING WATCHER")
        logger.info(f"{'='*80}\n")
        logger.info(f"🖥️  Hardware: Apple M4 Pro")
        logger.info(f"🚀 GPU: {'Metal (MPS)' if self.GPU_AVAILABLE else 'CPU only'}")
        logger.info(f"📊 Batch Size: {self.BATCH_SIZE}\n")
        logger.info(f"📋 Strategy:")
        logger.info(f"   1. Watch Process 1 (Super GMs from Chess.com)")
        logger.info(f"   2. Watch Process 2 (Top 1000 from Lichess)")
        logger.info(f"   3. Train when new data arrives")
        logger.info(f"   4. Repeat forever\n")
        logger.info(f"🎯 Model gets smarter continuously!\n")
        
        CHECK_INTERVAL = 60  # Check every minute
        
        try:
            while True:
                logger.info(f"👁️  Checking for new data...")
                
                if self.check_for_new_data():
                    logger.info(f"🎉 New data found! Starting training...")
                    self.train_model()
                else:
                    logger.info(f"💤 No new data, waiting {CHECK_INTERVAL}s...")
                
                time.sleep(CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info(f"\n⏹️  Stopped by user")
            logger.info(f"📊 Models trained: {self.state['models_trained']}")
            self.save_state()

def main():
    watcher = TrainingWatcher()
    watcher.run()

if __name__ == '__main__':
    main()
