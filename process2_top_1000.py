#!/usr/bin/env python3

"""
Anti-Stockfish Process 2: Top 1000 Collector
Source: Lichess
Strategy: ONE player at a time, 500 games per batch
"""

import requests
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [P2-TOP-1000] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process2_top_1000.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Top1000Collector:
    """Collect from Top 1000 on Lichess"""
    
    def __init__(self):
        self.data_dir = Path("neural_network/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_file = self.data_dir / "top_1000_dataset.jsonl"
        self.state_file = Path("process2_state.json")
        
        self.GAMES_PER_BATCH = 500
        self.load_state()
    
    def load_state(self):
        """Load state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
            logger.info(f"📊 Loaded: {len(self.state['top_players'])} players, {self.state['current_index']} processed")
        else:
            self.state = {
                'top_players': [],
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
    
    def fetch_top_players(self, count=1000):
        """Fetch top players from Lichess"""
        logger.info(f"📥 Fetching top {count} players from Lichess...")
        
        try:
            players = []
            
            # Fetch from multiple categories
            categories = ['bullet', 'blitz', 'rapid', 'classical', 'ultraBullet']
            
            for category in categories:
                if len(players) >= count:
                    break
                
                url = f"https://lichess.org/api/player/top/200/{category}"
                
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    for player in data.get('users', []):
                        username = player.get('username') or player.get('id')
                        if username and username not in players:
                            players.append(username)
                            if len(players) >= count:
                                break
                    
                    logger.info(f"  ✅ {category}: {len(data.get('users', []))} players")
                    time.sleep(1)  # Rate limiting between categories
                
                except Exception as e:
                    logger.warning(f"  ⚠️  {category}: {e}")
                    continue
            
            logger.info(f"✅ Fetched {len(players)} unique top players")
            return players[:count]
        
        except Exception as e:
            logger.error(f"❌ Error fetching players: {e}")
            return []
    
    def collect_from_lichess(self, username, max_games=500):
        """Collect games from Lichess"""
        logger.info(f"📥 Collecting {max_games} games from {username} (Lichess)...")
        
        try:
            url = f"https://lichess.org/api/games/user/{username}"
            params = {
                'max': max_games,
                'pgnInJson': True,
                'rated': True,
                'perfType': 'blitz,rapid,classical'
            }
            
            response = requests.get(url, params=params, timeout=120, stream=True)
            
            if response.status_code == 404:
                logger.warning(f"  ⚠️  {username}: User not found")
                return 0
            
            if response.status_code == 429:
                logger.warning(f"  ⚠️  {username}: Rate limited, waiting 60s...")
                time.sleep(60)
                return self.collect_from_lichess(username, max_games)
            
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
                logger.warning(f"  ⚠️  {username}: No games found")
                return 0
            
            # Save to dataset
            with open(self.dataset_file, 'a') as f:
                for pos in positions:
                    f.write(json.dumps(pos) + '\n')
            
            logger.info(f"  ✅ {username}: {games_count} games from Lichess")
            
            self.state['player_games'][username] = self.state['player_games'].get(username, 0) + games_count
            self.state['total_positions'] += games_count
            
            return games_count
        
        except Exception as e:
            logger.error(f"  ❌ {username}: {e}")
            return 0
    
    def run(self):
        """Main loop"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🌍 PROCESS 2: TOP 1000 COLLECTOR (Lichess)")
        logger.info(f"{'='*80}\n")
        logger.info(f"📊 Target: 1000 top players")
        logger.info(f"📊 Games per player: {self.GAMES_PER_BATCH}")
        logger.info(f"📊 Source: Lichess\n")
        
        try:
            # Fetch top players if not already done
            if not self.state['top_players']:
                self.state['top_players'] = self.fetch_top_players(1000)
                self.save_state()
            
            if not self.state['top_players']:
                logger.error("❌ No players to collect from")
                return
            
            logger.info(f"📊 Total players: {len(self.state['top_players'])}\n")
            
            while self.state['current_index'] < len(self.state['top_players']):
                idx = self.state['current_index']
                player = self.state['top_players'][idx]
                
                logger.info(f"\n{'='*80}")
                logger.info(f"👤 PLAYER {idx + 1}/{len(self.state['top_players'])}: {player}")
                logger.info(f"{'='*80}\n")
                
                games = self.collect_from_lichess(player, self.GAMES_PER_BATCH)
                
                if games > 0:
                    logger.info(f"✅ Progress: {idx + 1}/{len(self.state['top_players'])} complete")
                    logger.info(f"📊 Total games collected: {self.state['total_positions']:,}\n")
                else:
                    logger.warning(f"⏭️  Skipping {player}\n")
                
                self.state['current_index'] += 1
                self.save_state()
                
                time.sleep(5)  # Delay between players
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🎉 ALL {len(self.state['top_players'])} PLAYERS COMPLETE!")
            logger.info(f"{'='*80}\n")
            logger.info(f"✅ Total games collected: {self.state['total_positions']:,}\n")
        
        except KeyboardInterrupt:
            logger.info(f"\n⏹️  Stopped by user")
            self.save_state()
            sys.exit(0)

def main():
    collector = Top1000Collector()
    collector.run()

if __name__ == '__main__':
    main()
