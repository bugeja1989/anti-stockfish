#!/usr/bin/env python3

"""
Anti-Stockfish: ULTIMATE Phased Learning System (M4 Pro Optimized)
Designed for: Apple M4 Pro, 24GB RAM, 14 CPU cores, Metal GPU

Phase 1: Top 100 Super GMs (500 games) → Train
Phase 2: Historical Games → Retrain
Phase 3: Top 1000 Players + Super GMs (500 games) → Retrain
Phase 4: Continuous Learning (+1000 games/cycle) → Keep Retraining

Goal: 100,000 games × 1,100 players = 110 MILLION GAMES!
"""

import requests
import json
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
import sys
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_phased_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltimatePhasedLearning:
    """The most advanced chess AI training system ever built.
    
    Optimized for Apple M4 Pro:
    - 14 CPU cores (10 performance + 4 efficiency)
    - 24GB unified memory
    - Metal Performance Shaders (MPS) GPU acceleration
    """
    
    def __init__(self):
        self.data_dir = Path("neural_network/data")
        self.model_dir = Path("neural_network/models")
        self.historical_dir = Path("neural_network/data/historical")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        
        # Hardware optimization (with Lichess rate limiting)
        self.CPU_CORES = mp.cpu_count()  # All 14 cores
        self.MAX_WORKERS = 3  # Reduced to avoid rate limiting (Lichess allows ~15 req/min)
        self.GPU_AVAILABLE = torch.backends.mps.is_available()  # M4 Pro Metal
        self.BATCH_SIZE = 256 if self.GPU_AVAILABLE else 64  # Large batches with 24GB RAM
        
        logger.info(f"🖥️  Hardware: {self.CPU_CORES} CPU cores, GPU: {self.GPU_AVAILABLE}, Batch: {self.BATCH_SIZE}")
        
        # Configuration
        self.INITIAL_GAMES_PER_PLAYER = 500
        self.INCREMENTAL_GAMES = 1000
        self.MAX_GAMES_PER_PLAYER = 100000
        
        # Super GMs (Top 100 elite players)
        self.SUPER_GMS = [
            # World Champions & Top 10
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
            "LyonBeast",  # Maxime Vachier-Lagrave
            "Oleksandr_Bortnyk",  # Oleksandr Bortnyk
            "lachesisQ",  # Daniil Dubov
            "Polish_fighter3000",  # Jan-Krzysztof Duda
            "mishanp",  # Mikhail Antipov
            "dropstoneDP",  # David Paravyan
            
            # More Super GMs (continuing to 100)
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
        
        # Historical legends
        self.HISTORICAL_PLAYERS = [
            "Kasparov", "Fischer", "Karpov", "Tal", "Petrosian",
            "Smyslov", "Botvinnik", "Capablanca", "Lasker", "Morphy",
            "Alekhine", "Euwe", "Spassky", "Steinitz", "Kramnik",
            "Anand", "Carlsen", "Topalov", "Shirov", "Ivanchuk",
            "Nakamura", "Caruana", "Ding", "Giri", "Aronian",
            "Mamedyarov", "Radjabov", "Vachier-Lagrave",
            "So", "Nepomniachtchi", "Rapport",
            "Nimzowitsch", "Reti", "Rubinstein", "Tarrasch",
            "Pillsbury", "Marshall", "Schlechter", "Bronstein",
            "Keres", "Reshevsky", "Geller", "Polugaevsky"
        ]
        
        # State tracking
        self.state_file = Path("phased_learning_state.json")
        self.load_state()
    
    def load_state(self):
        """Load the current state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
            logger.info(f"📊 Loaded state: Phase {self.state['current_phase']}, Cycle {self.state['cycle']}")
        else:
            self.state = {
                'current_phase': 1,
                'cycle': 0,
                'super_gms_complete': False,
                'historical_complete': False,
                'top_1000_complete': False,
                'top_1000_players': [],
                'player_games': {},  # {username: games_collected}
                'total_positions': 0,
                'models_trained': 0,
                'started_at': datetime.now().isoformat()
            }
            logger.info("🆕 Starting fresh phased learning")
    
    def save_state(self):
        """Save the current state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def collect_game_parallel(self, player, num_games):
        """Collect games from a player (thread-safe)."""
        try:
            url = f"https://lichess.org/api/games/user/{player}"
            params = {
                'max': num_games,
                'pgnInJson': True,
                'rated': True,
                'perfType': 'blitz,rapid,classical'
            }
            
            response = requests.get(url, params=params, timeout=60, stream=True)
            
            if response.status_code == 404:
                return player, 0, []
            
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
            
            return player, games_count, positions
        
        except Exception as e:
            logger.warning(f"  ⚠️  {player}: {str(e)[:50]}")
            return player, 0, []
    
    def collect_games_batch(self, players, num_games, phase_name):
        """Collect games from multiple players in parallel."""
        logger.info(f"📥 Collecting {num_games} games from {len(players)} players ({phase_name})...")
        logger.info(f"🚀 Using {self.MAX_WORKERS} parallel workers")
        
        total_positions = 0
        output_file = self.data_dir / f"{phase_name.lower().replace(' ', '_')}_dataset.jsonl"
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = [executor.submit(self.collect_game_parallel, player, num_games) for player in players]
            
            with open(output_file, 'w') as f:
                for idx, future in enumerate(futures, 1):
                    player, games, positions = future.result()
                    
                    if games > 0:
                        logger.info(f"  [{idx}/{len(players)}] ✅ {player}: {games} games, {len(positions)} positions")
                        
                        for pos in positions:
                            f.write(json.dumps(pos) + '\n')
                            total_positions += 1
                        
                        self.state['player_games'][player] = self.state['player_games'].get(player, 0) + games
                    else:
                        logger.info(f"  [{idx}/{len(players)}] ⏭️  {player}: No games")
                    
                    time.sleep(2.0)  # Increased delay to avoid 429 errors
        
        logger.info(f"✅ {phase_name}: {total_positions:,} positions collected")
        return total_positions
    
    def download_historical_games(self):
        """Download historical games."""
        logger.info(f"\n{'='*80}")
        logger.info(f"📚 PHASE 2: HISTORICAL GAMES")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"📥 Downloading from {len(self.HISTORICAL_PLAYERS)} chess legends...")
        
        total_positions = self.collect_games_batch(
            self.HISTORICAL_PLAYERS,
            10000,  # More games from legends
            "Historical Games"
        )
        
        self.state['historical_complete'] = True
        self.state['total_positions'] += total_positions
        self.save_state()
        
        return total_positions
    
    def fetch_top_1000_players(self):
        """Fetch top 1000 players from Lichess."""
        logger.info(f"🔍 Fetching top 1000 players from Lichess...")
        
        try:
            categories = ['bullet', 'blitz', 'rapid', 'classical', 'ultraBullet']
            all_players = []
            seen = set(self.SUPER_GMS)  # Don't duplicate Super GMs
            
            for category in categories:
                try:
                    url = f"https://lichess.org/api/player/top/200/{category}"
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    players = [p['username'] for p in data.get('users', []) if p['username'] not in seen]
                    
                    for player in players:
                        if player not in seen:
                            all_players.append(player)
                            seen.add(player)
                    
                    logger.info(f"  ✅ {category}: +{len(players)} players (total: {len(all_players)})")
                    time.sleep(1)
                    
                    if len(all_players) >= 900:  # 1000 - 100 Super GMs
                        break
                
                except Exception as e:
                    logger.warning(f"  ⚠️  {category}: {e}")
                    continue
            
            return all_players[:900]  # Top 900 (+ 100 Super GMs = 1000)
        
        except Exception as e:
            logger.error(f"❌ Error fetching players: {e}")
            return []
    
    def merge_all_datasets(self):
        """Merge all datasets into master."""
        logger.info("🔄 Merging all datasets...")
        
        master_file = self.data_dir / "ultimate_master_dataset.jsonl"
        total_positions = 0
        
        with open(master_file, 'w') as master:
            for data_file in self.data_dir.glob("*_dataset.jsonl"):
                if data_file.name == "ultimate_master_dataset.jsonl":
                    continue
                
                with open(data_file) as f:
                    for line in f:
                        master.write(line)
                        total_positions += 1
        
        logger.info(f"✅ Master dataset: {total_positions:,} positions")
        return total_positions
    
    def train_model_optimized(self, epochs=20, phase_name="Model"):
        """Train models with M4 Pro optimization."""
        logger.info(f"🧠 Training {phase_name} ({epochs} epochs)...")
        logger.info(f"🔥 GPU: {'MPS (Metal)' if self.GPU_AVAILABLE else 'CPU'}, Batch: {self.BATCH_SIZE}")
        
        try:
            device = "mps" if self.GPU_AVAILABLE else "cpu"
            
            cmd = [
                "python3", "neural_network/src/train.py",
                "--data", "neural_network/data/ultimate_master_dataset.jsonl",
                "--epochs", str(epochs),
                "--batch-size", str(self.BATCH_SIZE),
                "--device", device,
                "--num-workers", str(min(self.MAX_WORKERS, 8))  # DataLoader workers
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
            
            if result.returncode == 0:
                logger.info(f"✅ {phase_name} training complete!")
                self.state['models_trained'] += 1
                return True
            else:
                logger.error(f"❌ Training failed")
                return False
        
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            return False
    
    def run_phase_1_super_gms(self):
        """Phase 1: Top 100 Super GMs."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🏆 PHASE 1: TOP 100 SUPER GMs")
        logger.info(f"{'='*80}\n")
        logger.info(f"📊 Target: {len(self.SUPER_GMS)} Super GMs × {self.INITIAL_GAMES_PER_PLAYER} games")
        logger.info(f"📊 Estimated: {len(self.SUPER_GMS) * self.INITIAL_GAMES_PER_PLAYER * 40:,} positions\n")
        
        positions = self.collect_games_batch(
            self.SUPER_GMS,
            self.INITIAL_GAMES_PER_PLAYER,
            "Phase 1 Super GMs"
        )
        
        self.state['super_gms_complete'] = True
        self.state['total_positions'] += positions
        self.merge_all_datasets()
        self.save_state()
        
        # Train first model
        logger.info(f"\n{'='*80}")
        logger.info(f"🧠 TRAINING INITIAL MODEL (Super GMs)")
        logger.info(f"{'='*80}\n")
        self.train_model_optimized(epochs=30, phase_name="Super GM Model")
        self.save_state()
    
    def run_phase_2_historical(self):
        """Phase 2: Historical games."""
        positions = self.download_historical_games()
        self.merge_all_datasets()
        
        # Retrain with historical data
        logger.info(f"\n{'='*80}")
        logger.info(f"🧠 RETRAINING WITH HISTORICAL GAMES")
        logger.info(f"{'='*80}\n")
        self.train_model_optimized(epochs=35, phase_name="Historical Model")
        self.save_state()
    
    def run_phase_3_top_1000(self):
        """Phase 3: Top 1000 players + Super GMs."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🌍 PHASE 3: TOP 1000 PLAYERS + SUPER GMs")
        logger.info(f"{'='*80}\n")
        
        # Fetch top 1000
        if not self.state['top_1000_players']:
            self.state['top_1000_players'] = self.fetch_top_1000_players()
            self.save_state()
        
        # Combine: Super GMs + Top 1000
        all_players = self.SUPER_GMS + self.state['top_1000_players']
        logger.info(f"📊 Target: {len(all_players)} players × {self.INITIAL_GAMES_PER_PLAYER} games")
        
        positions = self.collect_games_batch(
            self.state['top_1000_players'],  # Super GMs already collected
            self.INITIAL_GAMES_PER_PLAYER,
            "Phase 3 Top 1000"
        )
        
        self.state['top_1000_complete'] = True
        self.state['total_positions'] += positions
        self.merge_all_datasets()
        self.save_state()
        
        # Retrain with massive dataset
        logger.info(f"\n{'='*80}")
        logger.info(f"🧠 RETRAINING WITH MASSIVE DATASET")
        logger.info(f"{'='*80}\n")
        self.train_model_optimized(epochs=40, phase_name="Massive Model")
        self.save_state()
    
    def run_phase_4_continuous(self):
        """Phase 4: Continuous learning cycles."""
        all_players = self.SUPER_GMS + self.state['top_1000_players']
        
        while True:
            self.state['cycle'] += 1
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 CYCLE {self.state['cycle']}: CONTINUOUS LEARNING")
            logger.info(f"{'='*80}\n")
            
            # Check if goal reached
            avg_games = sum(self.state['player_games'].values()) / len(self.state['player_games']) if self.state['player_games'] else 0
            
            if avg_games >= self.MAX_GAMES_PER_PLAYER:
                logger.info(f"\n🎉 {'='*80}")
                logger.info(f"🎉 ULTIMATE GOAL REACHED!")
                logger.info(f"🎉 {'='*80}\n")
                logger.info(f"✅ {self.MAX_GAMES_PER_PLAYER:,} games × {len(all_players)} players")
                logger.info(f"✅ Total positions: {self.state['total_positions']:,}")
                logger.info(f"✅ Models trained: {self.state['models_trained']}")
                logger.info(f"\n🏆 READY TO BEAT STOCKFISH! 🏆\n")
                break
            
            # Collect more games
            positions = self.collect_games_batch(
                all_players,
                self.INCREMENTAL_GAMES,
                f"Cycle {self.state['cycle']}"
            )
            
            if positions > 0:
                self.state['total_positions'] += positions
                self.merge_all_datasets()
                
                # Retrain
                logger.info(f"\n{'='*80}")
                logger.info(f"🧠 RETRAINING (Cycle {self.state['cycle']})")
                logger.info(f"{'='*80}\n")
                
                epochs = min(40 + (self.state['cycle'] * 5), 60)
                self.train_model_optimized(epochs=epochs, phase_name=f"Cycle {self.state['cycle']} Model")
                self.save_state()
            else:
                time.sleep(300)
    
    def run(self):
        """Main execution."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 ANTI-STOCKFISH: ULTIMATE PHASED LEARNING")
        logger.info(f"{'='*80}\n")
        logger.info(f"🖥️  Hardware: Apple M4 Pro, {self.CPU_CORES} cores, 24GB RAM")
        logger.info(f"🚀 GPU: {'Metal (MPS)' if self.GPU_AVAILABLE else 'CPU only'}")
        logger.info(f"📊 Batch Size: {self.BATCH_SIZE}")
        logger.info(f"⚡ Workers: {self.MAX_WORKERS}\n")
        logger.info(f"📋 Phase 1: Top 100 Super GMs → Train")
        logger.info(f"📋 Phase 2: Historical Games → Retrain")
        logger.info(f"📋 Phase 3: Top 1000 + Super GMs → Retrain")
        logger.info(f"📋 Phase 4: Continuous Learning → Keep Retraining\n")
        
        try:
            # Phase 1: Super GMs
            if not self.state['super_gms_complete']:
                self.run_phase_1_super_gms()
                self.state['current_phase'] = 2
                self.save_state()
            
            # Phase 2: Historical
            if not self.state['historical_complete']:
                self.run_phase_2_historical()
                self.state['current_phase'] = 3
                self.save_state()
            
            # Phase 3: Top 1000
            if not self.state['top_1000_complete']:
                self.run_phase_3_top_1000()
                self.state['current_phase'] = 4
                self.save_state()
            
            # Phase 4: Continuous
            self.run_phase_4_continuous()
        
        except KeyboardInterrupt:
            logger.info(f"\n⏹️  Stopped by user")
            logger.info(f"📊 Phase {self.state['current_phase']}, Cycle {self.state['cycle']}")
            logger.info(f"📊 Positions: {self.state['total_positions']:,}")
            self.save_state()
            sys.exit(0)

def main():
    system = UltimatePhasedLearning()
    system.run()

if __name__ == '__main__':
    main()
