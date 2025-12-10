#!/usr/bin/env python3

"""
Anti-Stockfish: Optimized Training Script for Apple M4 Pro
- Metal Performance Shaders (MPS) GPU acceleration
- Streaming Dataset (IterableDataset) for low memory usage
- Large batch sizes (1024+) for max throughput
- Optimized for Apple Silicon
- Supports Checkpointing & Resuming
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import json
import logging
import argparse
from pathlib import Path
import sys
import itertools
import os

# Import the model
sys.path.append(str(Path(__file__).parent))
from model import ChaosModule, SacrificeModule, board_to_tensor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamingChessDataset(IterableDataset):
    """Memory-efficient streaming dataset for large files."""
    
    def __init__(self, data_file):
        self.data_file = data_file
        
    def __iter__(self):
        """Yields batches of data directly from file without loading everything to RAM."""
        with open(self.data_file) as f:
            for line in f:
                try:
                    pos = json.loads(line)
                    
                    # Convert position to tensor
                    board_tensor = board_to_tensor(pos['fen'])
                    
                    # Labels
                    label_map = {'normal': 0, 'complex': 1, 'sacrifice': 2}
                    label = label_map.get(pos.get('label', 'normal'), 0)
                    
                    # Evaluation (normalized)
                    eval_score = pos.get('eval', 0.0)
                    eval_normalized = max(-1.0, min(1.0, eval_score / 10.0))
                    
                    yield {
                        'board': board_tensor,
                        'label': torch.tensor(label, dtype=torch.long),
                        'eval': torch.tensor(eval_normalized, dtype=torch.float32),
                        'chaos': torch.tensor(1.0 if label == 2 else 0.5 if label == 1 else 0.0, dtype=torch.float32)
                    }
                except:
                    continue

def train_epoch(chaos_model, sacrifice_model, dataloader, optimizer_chaos, optimizer_sac, device, total_batches):
    """Train for one epoch."""
    chaos_model.train()
    sacrifice_model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        board = batch['board'].to(device)
        label = batch['label'].to(device)
        eval_target = batch['eval'].to(device)
        chaos_target = batch['chaos'].to(device)
        
        # Chaos Module forward
        optimizer_chaos.zero_grad()
        policy, value, chaos = chaos_model(board)
        
        # Chaos loss
        policy_loss = nn.CrossEntropyLoss()(policy, label)
        value_loss = nn.MSELoss()(value.squeeze(), eval_target)
        chaos_loss = nn.MSELoss()(chaos.squeeze(), chaos_target)
        
        total_chaos_loss = policy_loss + value_loss + chaos_loss
        total_chaos_loss.backward()
        optimizer_chaos.step()
        
        # Sacrifice Module forward
        optimizer_sac.zero_grad()
        sacrifice_prob = sacrifice_model(board)
        sacrifice_target = (label == 2).float()
        sacrifice_loss = nn.BCELoss()(sacrifice_prob.squeeze(), sacrifice_target)
        sacrifice_loss.backward()
        optimizer_sac.step()
        
        total_loss += (total_chaos_loss.item() + sacrifice_loss.item())
        num_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            percent = ((batch_idx + 1) / total_batches) * 100
            logger.info(f"  Batch {batch_idx + 1}/{total_batches} ({percent:.1f}%) | Loss: {avg_loss:.8f}")
    
    return total_loss / max(1, num_batches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--device', default='mps', help='Device (mps, cuda, cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint if available')
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("🔥 Using Metal Performance Shaders (MPS) GPU!")
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("🔥 Using CUDA GPU!")
    else:
        device = torch.device('cpu')
        logger.info("💻 Using CPU")
    
    logger.info(f"⚙️  Batch size: {args.batch_size}")
    logger.info(f"⚙️  Workers: {args.num_workers}")
    logger.info(f"⚙️  Epochs: {args.epochs}")
    
    # Load dataset (Streaming)
    dataset = StreamingChessDataset(args.data)
    
    # Calculate total batches (approximate based on file line count)
    total_lines = 0
    try:
        with open(args.data) as f:
            for _ in f: total_lines += 1
    except:
        total_lines = 1000000 # Fallback
        
    total_batches = total_lines // args.batch_size
    logger.info(f"📊 Total positions: {total_lines:,}")
    logger.info(f"📦 Total batches per epoch: {total_batches:,}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False  # Disabled for MPS stability
    )
    
    # Models
    chaos_model = ChaosModule().to(device)
    sacrifice_model = SacrificeModule().to(device)
    
    # Optimizers
    optimizer_chaos = optim.Adam(chaos_model.parameters(), lr=0.001)
    optimizer_sac = optim.Adam(sacrifice_model.parameters(), lr=0.001)
    
    # Checkpoint directory
    model_dir = Path("neural_network/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    chaos_path = model_dir / "chaos_module.pth"
    sac_path = model_dir / "sacrifice_module.pth"
    
    # Resume logic
    start_epoch = 0
    if args.resume:
        if chaos_path.exists() and sac_path.exists():
            logger.info("🔄 Found existing models. Resuming training...")
            try:
                chaos_model.load_state_dict(torch.load(chaos_path, map_location=device))
                sacrifice_model.load_state_dict(torch.load(sac_path, map_location=device))
                logger.info("✅ Models loaded successfully!")
            except Exception as e:
                logger.error(f"⚠️ Failed to load models: {e}. Starting from scratch.")
        else:
            logger.info("ℹ️  No existing models found. Starting fresh.")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🧠 TRAINING STARTED (STREAMING MODE)")
    logger.info(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        avg_loss = train_epoch(
            chaos_model,
            sacrifice_model,
            dataloader,
            optimizer_chaos,
            optimizer_sac,
            device,
            total_batches
        )
        
        logger.info(f"✅ Epoch {epoch + 1} complete! Avg Loss: {avg_loss:.8f}")
        
        # Save checkpoint after EVERY epoch
        torch.save(chaos_model.state_dict(), chaos_path)
        torch.save(sacrifice_model.state_dict(), sac_path)
        logger.info(f"💾 Checkpoint saved to {model_dir}\n")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ TRAINING COMPLETE!")
    logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
