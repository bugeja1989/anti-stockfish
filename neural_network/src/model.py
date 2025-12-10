#!/usr/bin/env python3

"""
Anti-Stockfish: Neural Network Models
- Chaos Module: Position evaluation with chaos/sacrifice bias
- Sacrifice Module: Detects sacrificial opportunities
"""

import torch
import torch.nn as nn
import chess
import numpy as np

def board_to_tensor(fen):
    """
    Convert FEN position to tensor representation.
    Returns: [12, 8, 8] tensor (12 piece types × 8×8 board)
    """
    board = chess.Board(fen)
    
    # 12 channels: 6 piece types × 2 colors
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)  # Flip vertically
            col = square % 8
            
            channel = piece_idx[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            
            tensor[channel, row, col] = 1.0
    
    return torch.from_numpy(tensor)


class ChaosModule(nn.Module):
    """
    Chaos Module: Evaluates positions with bias toward complex/sacrificial play
    
    Architecture:
    - Input: 12×8×8 board representation
    - Conv layers: Extract spatial features
    - Three heads:
      1. Policy head: Which move to play
      2. Value head: Position evaluation
      3. Chaos head: Complexity/sacrifice score
    """
    
    def __init__(self):
        super(ChaosModule, self).__init__()
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Policy head (move selection)
        self.policy_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.policy_fc = nn.Linear(128 * 8 * 8, 4096)  # All possible moves
        
        # Value head (position evaluation)
        self.value_conv = nn.Conv2d(256, 64, kernel_size=1)
        self.value_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Chaos head (complexity/sacrifice score)
        self.chaos_conv = nn.Conv2d(256, 64, kernel_size=1)
        self.chaos_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.chaos_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Forward pass
        Input: [batch, 12, 8, 8]
        Output: (policy, value, chaos)
        """
        # Shared convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Policy head
        policy = self.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = self.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = self.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # [-1, 1]
        
        # Chaos head
        chaos = self.relu(self.chaos_conv(x))
        chaos = chaos.view(chaos.size(0), -1)
        chaos = self.relu(self.chaos_fc1(chaos))
        chaos = torch.sigmoid(self.chaos_fc2(chaos))  # [0, 1]
        
        return policy, value, chaos


class SacrificeModule(nn.Module):
    """
    Sacrifice Module: Specialized network for detecting sacrificial opportunities
    
    Smaller, faster network focused on tactical patterns
    """
    
    def __init__(self):
        super(SacrificeModule, self).__init__()
        
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Sacrifice detection head
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)  # Sacrifice score [0, 1]
    
    def forward(self, x):
        """
        Forward pass
        Input: [batch, 12, 8, 8]
        Output: sacrifice_score [0, 1]
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x


def create_models(device='cpu'):
    """Create and initialize both models"""
    chaos_module = ChaosModule().to(device)
    sacrifice_module = SacrificeModule().to(device)
    
    return chaos_module, sacrifice_module


if __name__ == '__main__':
    # Test the models
    print("Testing Anti-Stockfish Neural Networks...")
    
    # Test board_to_tensor
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = board_to_tensor(fen)
    print(f"✅ board_to_tensor: {tensor.shape}")
    
    # Test ChaosModule
    chaos = ChaosModule()
    batch = tensor.unsqueeze(0)  # Add batch dimension
    policy, value, chaos_score = chaos(batch)
    print(f"✅ ChaosModule:")
    print(f"   Policy: {policy.shape}")
    print(f"   Value: {value.shape} = {value.item():.3f}")
    print(f"   Chaos: {chaos_score.shape} = {chaos_score.item():.3f}")
    
    # Test SacrificeModule
    sacrifice = SacrificeModule()
    sacrifice_score = sacrifice(batch)
    print(f"✅ SacrificeModule:")
    print(f"   Sacrifice: {sacrifice_score.shape} = {sacrifice_score.item():.3f}")
    
    print("\n🎉 All models working!")
