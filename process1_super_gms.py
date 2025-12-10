#!/usr/bin/env python3

"""
Anti-Stockfish Process 1: Super GMs Collector
Source: Chess.com (PREFERRED)
Strategy: ONE player at a time, 500 games per batch
"""

import requests
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import sys
import chess.pgn
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [P1-SUPER-GMS] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process1_super_gms.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SuperGMsCollector:
    """Collect from Super GMs on Chess.com"""
    
    def __init__(self):
        self.data_dir = Path("neural_network/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_file = self.data_dir / "super_gms_dataset.jsonl"
        self.state_file = Path("process1_state.json")
        
        # Chess.com Super GMs (verified usernames)
        self.SUPER_GMS = [
            "MagnusCarlsen",  # Magnus Carlsen
            "Hikaru",  # Hikaru Nakamura
            "FabianoCaruana",  # Fabiano Caruana
            "DanielNaroditsky",  # Daniel Naroditsky
            "GothamChess",  # Levy Rozman
            "GMHess",  # Robert Hess
            "GMWSO",  # Eric Hansen
            "BotezLive",  # Alexandra Botez
            "annacramling",  # Anna Cramling
            "GMBenjaminFinegold",  # Benjamin Finegold
            "ChessNetwork",  # Jerry
            "agadmator",  # Antonio Radic
            "DanyaTheGM",  # Daniil Dubov
            "Ginger_GM",  # Simon Williams
            "IMRosen",  # Eric Rosen
            "ChessBrah",  # Aman Hambleton
            "GMNaroditsky",  # Daniel alt
            "penguingm1",  # Andrew Tang
            "nihalsarin",  # Nihal Sarin
            "GMBENJAMINBOK",  # Benjamin Bok
            "LyonBeast",  # MVL
            "Polish_fighter3000",  # Jan-Krzysztof Duda
            "mishanp",  # Mikhail Antipov
            "dropstoneDP",  # David Paravyan
            "Jospem",  # Jose Eduardo Martinez
            "BillieKimbah",  # Billie Kimber
            "Lovlas",  # Lovlas
            "GM_dmitrij",  # Dmitrij
            "RealDavidNavara",  # David Navara
            "Lu_Shanglei",  # Lu Shanglei
            "Kacparov",  # Kacper Piorun
            "AlexTriapishko",  # Alex Triapishko
            "Zhalmakhanov_R",  # Rinat Zhalmakhanov
            "AbasovN",  # Nijat Abasov
            "RakitinD",  # Dmitry Rakitin
            "Pap-G",  # Gabor Pap
            "Grandelicious",  # Grandmaster
            "TargetGM",  # Target GM
            "Kirill_Klyukin",  # Kirill Klyukin
            "Rakhmanov_Aleksandr",  # Aleksandr Rakhmanov
            "Ramil_Faizrakhmanov",  # Ramil Faizrakhmanov
            "DrawDenied_Twitch",  # DrawDenied
            "Aristarch_Kukuev",  # Aristarch Kukuev
            "BelyakovBogdan",  # Bogdan Belyakov
            "Tomberg_Dmitry",  # Dmitry Tomberg
            "Bocharov-Ivan",  # Ivan Bocharov
            "EminovOrkhan",  # Orkhan Eminov
            "AtalikS",  # Suat Atalik
            "Alexander_Abramov",  # Alexander Abramov
            "Vlad_Lazarev79",  # Vlad Lazarev
            "AlexanderMitrofanov",  # Alexander Mitrofanov
            "ZakharElyashevich",  # Zakhar Elyashevich
            "Dmitriychesser",  # Dmitriy
            "Rechits_Maxim",  # Maxim Rechits
            "Sasha_Lopatnikova",  # Sasha Lopatnikova
            "MaximEvdo",  # Maxim Evdokimov
            "Saif_2004_JOR",  # Saif
            "SavvaVetokhin2009",  # Savva Vetokhin
            "Matviy2009",  # Matviy
            "Oleg_Chess14",  # Oleg
            "Boburjon007",  # Boburjon
            "Yakov25",  # Yakov
            "Suleyman_Chess06",  # Suleyman
            "Soham0705",  # Soham
            "Absolute_Power",  # Absolute Power
            "ChessTheory64",  # Chess Theory
            "AnalysisKing",  # Analysis King
            "CoolChessSchool",  # Cool Chess School
            "Unknown_Master459",  # Unknown Master
            "TechnicalTrader",  # Technical Trader
            "BackyardProfessor",  # Backyard Professor
            "Franklin_Lewis",  # Franklin Lewis
            "Champion_Reborn",  # Champion Reborn
            "WolfWinner",  # Wolf Winner
            "ConwyCastle",  # Conwy Castle
            "GoTheDistance",  # Go The Distance
            "TacticalMonkey",  # Tactical Monkey
            "ConquerorsHaki",  # Conquerors Haki
            "ReadySetGains",  # Ready Set Gains
            "FieldTactics",  # Field Tactics
            "Unbroken_Warrior",  # Unbroken Warrior
            "Haunting_Games",  # Haunting Games
            "ChessInsomniac",  # Chess Insomniac
            "TheGoatOfOpenings",  # The Goat Of Openings
            "DeepDutch",  # Deep Dutch
            "Schachstratege",  # Schachstratege
            "Raehgalchess",  # Raehgal
            "Rhythmofmind",  # Rhythm of Mind
            "understable",  # Understable
            "somerapidplayer",  # Some Rapid Player
            "Nosporchess",  # Nospor
            "DestineCrow444",  # Destine Crow
            "MoonlightAnomaly",  # Moonlight Anomaly
            "Neutralizerr",  # Neutralizer
            "visualdennis",  # Visual Dennis
            "PawnInTraining",  # Pawn In Training
            "interestinglandscape",  # Interesting Landscape
            "MangoMustardSixSeven",  # Mango Mustard
            "NowArcher",  # Now Archer
            "Breathing_water",  # Breathing Water
            "alien_from_the_moon",  # Alien From The Moon
        ]
        
        self.GAMES_PER_BATCH = 500
        self.load_state()
    
    def load_state(self):
        """Load state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
            logger.info(f"📊 Loaded: Player {self.state['current_index'] + 1}/{len(self.SUPER_GMS)}")
        else:
            self.state = {
                'current_index': 0,
                'player_games': {},
                'total_positions': 0,
                'started_at': datetime.now().isoformat()
            }
            logger.info("🆕 Starting fresh")
    
    def save_state(self):
        """Save state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def collect_from_chesscom(self, username, max_games=500):
        """Collect games from Chess.com"""
        logger.info(f"📥 Collecting {max_games} games from {username} (Chess.com)...")
        
        # Chess.com requires User-Agent header
        headers = {
            'User-Agent': 'Anti-Stockfish Chess Bot (https://github.com/bugeja1989/anti-stockfish)'
        }
        
        try:
            # Get player's archives
            archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
            response = requests.get(archives_url, headers=headers, timeout=30)
            
            if response.status_code == 404:
                logger.warning(f"  ⚠️  {username}: User not found on Chess.com")
                return 0
            
            response.raise_for_status()
            archives = response.json().get('archives', [])
            
            if not archives:
                logger.warning(f"  ⚠️  {username}: No game archives")
                return 0
            
            # Collect from recent archives
            positions = []
            games_collected = 0
            
            for archive_url in reversed(archives[-6:]):  # Last 6 months
                if games_collected >= max_games:
                    break
                
                time.sleep(2)  # Rate limiting
                
                try:
                    archive_response = requests.get(archive_url, headers=headers, timeout=30)
                    archive_response.raise_for_status()
                    
                    games = archive_response.json().get('games', [])
                    
                    for game in games:
                        if games_collected >= max_games:
                            break
                        
                        # Only rated games
                        if game.get('rated'):
                            positions.append(game)
                            games_collected += 1
                
                except Exception as e:
                    logger.warning(f"  ⚠️  Archive error: {e}")
                    continue
            
            if games_collected == 0:
                logger.warning(f"  ⚠️  {username}: No games collected")
                return 0
            
            # Save to dataset
            with open(self.dataset_file, 'a') as f:
                for pos in positions:
                    f.write(json.dumps(pos) + '\n')
            
            logger.info(f"  ✅ {username}: {games_collected} games from Chess.com")
            
            self.state['player_games'][username] = self.state['player_games'].get(username, 0) + games_collected
            self.state['total_positions'] += games_collected
            
            return games_collected
        
        except Exception as e:
            logger.error(f"  ❌ {username}: {e}")
            return 0
    
    def run(self):
        """Main loop"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🏆 PROCESS 1: SUPER GMS COLLECTOR (Chess.com)")
        logger.info(f"{'='*80}\n")
        logger.info(f"📊 Total Super GMs: {len(self.SUPER_GMS)}")
        logger.info(f"📊 Games per player: {self.GAMES_PER_BATCH}")
        logger.info(f"📊 Source: Chess.com (PREFERRED)\n")
        
        try:
            while self.state['current_index'] < len(self.SUPER_GMS):
                idx = self.state['current_index']
                player = self.SUPER_GMS[idx]
                
                logger.info(f"\n{'='*80}")
                logger.info(f"👤 PLAYER {idx + 1}/{len(self.SUPER_GMS)}: {player}")
                logger.info(f"{'='*80}\n")
                
                games = self.collect_from_chesscom(player, self.GAMES_PER_BATCH)
                
                if games > 0:
                    logger.info(f"✅ Progress: {idx + 1}/{len(self.SUPER_GMS)} complete")
                    logger.info(f"📊 Total positions: {self.state['total_positions']:,}\n")
                else:
                    logger.warning(f"⏭️  Skipping {player}\n")
                
                self.state['current_index'] += 1
                self.save_state()
                
                time.sleep(5)  # Delay between players
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🎉 ALL {len(self.SUPER_GMS)} SUPER GMS COMPLETE!")
            logger.info(f"{'='*80}\n")
            logger.info(f"✅ Total positions: {self.state['total_positions']:,}\n")
        
        except KeyboardInterrupt:
            logger.info(f"\n⏹️  Stopped by user")
            self.save_state()
            sys.exit(0)

def main():
    collector = SuperGMsCollector()
    collector.run()

if __name__ == '__main__':
    main()
