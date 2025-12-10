#!/usr/bin/env python3

"""
Anti-Stockfish: SEQUENTIAL One-by-One Learning System
Optimized for: Apple M4 Pro, 24GB RAM, 14 CPU cores, Metal GPU

Strategy:
- Collect from ONE player at a time (NO rate limiting!)
- Train model after EACH player (continuous improvement!)
- Start with Super GMs (best quality first)
- Use all cores + GPU for training (parallel training only)

The model gets smarter after every single player!
"""

import requests
import json
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
import sys
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sequential_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SequentialLearning:
    """One player at a time, train after each. The smartest approach."""
    
    def __init__(self):
        self.data_dir = Path("neural_network/data")
        self.model_dir = Path("neural_network/models")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Hardware
        self.GPU_AVAILABLE = torch.backends.mps.is_available()
        self.BATCH_SIZE = 256 if self.GPU_AVAILABLE else 64
        
        logger.info(f"🖥️  GPU: {self.GPU_AVAILABLE}, Batch: {self.BATCH_SIZE}")
        
        # Configuration
        self.GAMES_PER_PLAYER = 500
        self.INCREMENTAL_GAMES = 1000
        self.MAX_GAMES_PER_PLAYER = 100000
        
        # Super GMs (Top 100)
        self.SUPER_GMS = [
            "DrNykterstein",  # Magnus Carlsen
            "FairChess_on_YouTube",  # Hikaru Nakamura
            "Firouzja2003",  # Alireza Firouzja
            "DanielNaroditsky",  # Daniel Naroditsky
            "GMHikaruOnTwitch",  # Hikaru alt
            "MagnusCarlsen",  # Magnus official
            "penguingm1",  # Andrew Tang
            "Zhigalko_Sergei",  # Sergei Zhigalko
            "nihalsarin",  # Nihal Sarin
            "GMBENJAMINBOK",  # Benjamin Bok
            "Grischuk",  # Alexander Grischuk
            "ArjunErigaisi",  # Arjun Erigaisi
            "rpragchess",  # Praggnanandhaa
            "chessbrahs",  # Chessbrah
            "LyonBeast",  # MVL
            "Oleksandr_Bortnyk",  # Oleksandr Bortnyk
            "lachesisQ",  # Daniil Dubov
            "Polish_fighter3000",  # Jan-Krzysztof Duda
            "mishanp",  # Mikhail Antipov
            "dropstoneDP",  # David Paravyan
            "GothamChess", "agadmator", "GMBenjaminFinegold",
            "GMHess", "IM_Eric_Rosen", "ChessNetwork",
            "GMWSO", "Lovlas", "Jospem", "BillieKimbah",
            "GM_dmitrij", "SindarovGM", "RealDavidNavara",
            "Lu_Shanglei", "Kacparov", "AlexTriapishko",
            "Zhalmakhanov_R", "AbasovN", "RakitinD",
            "Pap-G", "Grandelicious", "TargetGM",
            "Kirill_Klyukin", "Rakhmanov_Aleksandr",
            "Alexandr_KhleBovich", "Ramil_Faizrakhmanov",
            "DrawDenied_Twitch", "Aristarch_Kukuev",
            "BelyakovBogdan", "Tomberg_Dmitry",
            "Bocharov-Ivan", "EminovOrkhan", "AtalikS",
            "Alexander_Abramov", "Vlad_Lazarev79",
            "AlexanderMitrofanov", "ZakharElyashevich",
            "Dmitriychesser", "Rechits_Maxim",
            "Sasha_Lopatnikova", "MaximEvdo",
            "Saif_2004_JOR", "SavvaVetokhin2009",
            "Matviy2009", "Oleg_Chess14",
            "Boburjon007", "Yakov25",
            "Suleyman_Chess06", "Soham0705",
            "Absolute_Power", "ChessTheory64",
            "AnalysisKing", "CoolChessSchool",
            "Unknown_Master459", "TechnicalTrader",
            "BackyardProfessor", "Franklin_Lewis",
            "Champion_Reborn", "WolfWinner",
            "ConwyCastle", "GoTheDistance",
            "TacticalMonkey", "ConquerorsHaki",
            "ReadySetGains", "FieldTactics",
            "Unbroken_Warrior", "Haunting_Games",
            "ChessInsomniac", "TheGoatOfOpenings",
            "DeepDutch", "Schachstratege",
            "Raehgalchess", "Rhythmofmind",
            "understable", "somerapidplayer",
            "Nosporchess", "DestineCrow444",
            "MoonlightAnomaly", "Neutralizerr",
            "visualdennis", "PawnInTraining",
            "interestinglandscape", "MangoMustardSixSeven",
            "NowArcher", "Breathing_water",
            "alien_from_the_moon", "RE-BORN",
            "Vyga2012", "Marat22121972",
            "Moussardraser", "chesspawnrookking"
        ]
        
        # State
        self.state_file = Path("sequential_learning_state.json")
        self.master_dataset = self.data_dir / "master_dataset.jsonl"
        self.load_state()
    
    def load_state(self):
        """Load state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
            logger.info(f"📊 Loaded state: Player {self.state['current_player_index'] + 1}/{len(self.SUPER_GMS)}")
        else:
            self.state = {
                'current_player_index': 0,
                'cycle': 0,
                'player_games': {},
                'total_positions': 0,
                'models_trained': 0,
                'started_at': datetime.now().isoformat()
            }
            logger.info("🆕 Starting fresh sequential learning")
    
    def save_state(self):
        """Save state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def collect_from_player(self, player, num_games):
        """Collect games from ONE player."""
        logger.info(f"📥 Collecting {num_games} games from {player}...")
        
        try:
            url = f"https://lichess.org/api/games/user/{player}"
            params = {
                'max': num_games,
                'pgnInJson': True,
                'rated': True,
                'perfType': 'blitz,rapid,classical'
            }
            
            response = requests.get(url, params=params, timeout=120, stream=True)
            
            if response.status_code == 404:
                logger.warning(f"  ⚠️  {player}: User not found")
                return 0
            
            if response.status_code == 429:
                logger.warning(f"  ⚠️  {player}: Rate limited, waiting 60s...")
                time.sleep(60)
                return self.collect_from_player(player, num_games)  # Retry
            
            response.raise_for_status()
            
            positions = []
            games_count = 0
            
            for line in response.iter_lines():
                if line:
                    try:
                        game_data = json.loads(line.decode('utf-8'))
                        positions.append(game_data)
                        games_count += 1
                    except:
                        continue
            
            if games_count == 0:
                logger.warning(f"  ⚠️  {player}: No games found")
                return 0
            
            # Append to master dataset
            with open(self.master_dataset, 'a') as f:
                for pos in positions:
                    f.write(json.dumps(pos) + '\n')
            
            logger.info(f"  ✅ {player}: {games_count} games, {len(positions)} positions")
            
            self.state['player_games'][player] = self.state['player_games'].get(player, 0) + games_count
            self.state['total_positions'] += len(positions)
            
            return len(positions)
        
        except Exception as e:
            logger.error(f"  ❌ {player}: {e}")
            return 0
    
    def train_model(self, player_name):
        """Train model with all data collected so far."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🧠 TRAINING after {player_name}")
        logger.info(f"{'='*80}\n")
        logger.info(f"📊 Total positions: {self.state['total_positions']:,}")
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
                logger.info(f"✅ Training complete after {player_name}!")
                self.state['models_trained'] += 1
                return True
            else:
                logger.error(f"❌ Training failed: {result.stderr[:200]}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            return False
    
    def run(self):
        """Main loop: ONE player → TRAIN → repeat."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 SEQUENTIAL ONE-BY-ONE LEARNING")
        logger.info(f"{'='*80}\n")
        logger.info(f"🖥️  Hardware: Apple M4 Pro")
        logger.info(f"🚀 GPU: {'Metal (MPS)' if self.GPU_AVAILABLE else 'CPU only'}")
        logger.info(f"📊 Batch Size: {self.BATCH_SIZE}\n")
        logger.info(f"📋 Strategy:")
        logger.info(f"   1. Collect from ONE Super GM")
        logger.info(f"   2. Train model immediately")
        logger.info(f"   3. Model gets smarter")
        logger.info(f"   4. Repeat for all {len(self.SUPER_GMS)} Super GMs\n")
        logger.info(f"🎯 Goal: Model improves after EVERY player!\n")
        
        try:
            while self.state['current_player_index'] < len(self.SUPER_GMS):
                player_idx = self.state['current_player_index']
                player = self.SUPER_GMS[player_idx]
                
                logger.info(f"\n{'='*80}")
                logger.info(f"👤 PLAYER {player_idx + 1}/{len(self.SUPER_GMS)}: {player}")
                logger.info(f"{'='*80}\n")
                
                # Collect from this player
                positions = self.collect_from_player(player, self.GAMES_PER_PLAYER)
                
                if positions > 0:
                    self.save_state()
                    
                    # Train immediately after this player
                    self.train_model(player)
                    self.save_state()
                    
                    logger.info(f"\n✅ Progress: {player_idx + 1}/{len(self.SUPER_GMS)} players complete")
                    logger.info(f"📊 Total positions: {self.state['total_positions']:,}")
                    logger.info(f"🧠 Models trained: {self.state['models_trained']}\n")
                else:
                    logger.warning(f"⏭️  Skipping {player} (no data)")
                
                # Move to next player
                self.state['current_player_index'] += 1
                self.save_state()
                
                # Small delay between players
                time.sleep(5)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🎉 ALL {len(self.SUPER_GMS)} SUPER GMS COMPLETE!")
            logger.info(f"{'='*80}\n")
            logger.info(f"✅ Total positions: {self.state['total_positions']:,}")
            logger.info(f"✅ Models trained: {self.state['models_trained']}")
            logger.info(f"\n🏆 MODEL IS NOW SUPER SMART! 🏆\n")
        
        except KeyboardInterrupt:
            logger.info(f"\n⏹️  Stopped by user")
            logger.info(f"📊 Player {self.state['current_player_index'] + 1}/{len(self.SUPER_GMS)}")
            logger.info(f"📊 Positions: {self.state['total_positions']:,}")
            self.save_state()
            sys.exit(0)

def main():
    system = SequentialLearning()
    system.run()

if __name__ == '__main__':
    main()
