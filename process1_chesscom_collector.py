#!/usr/bin/env python3

"""
Anti-Stockfish Process 1: Ultimate Chess.com Collector
GOAL: 100,000,000+ games and positions
Strategy: Continuously cycle through all categories, expanding player pool
"""

import requests
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [P1-COLLECTOR] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process1_chesscom.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MegaChessComCollector:
    """Aggressive collector targeting 100M+ positions"""
    
    def __init__(self):
        self.data_dir = Path("neural_network/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_file = self.data_dir / "chesscom_master_dataset.jsonl"
        self.state_file = Path("process1_state.json")
        
        # Rate limiting
        self.REQUEST_DELAY = 0.3  # 300ms between requests
        self.last_request_time = 0
        
        self.load_state()
    
    def load_state(self):
        """Load collection state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
                self.state = {
                    'total_games': data.get('total_games', 0),
                    'total_positions': data.get('total_positions', 0),
                    'players_processed': set(data.get('players_processed', [])),
                    'collection_rounds': data.get('collection_rounds', 0),
                    'last_update': data.get('last_update')
                }
            logger.info(f"📊 Loaded: {self.state['total_games']:,} games, {self.state['total_positions']:,} positions, {len(self.state['players_processed']):,} players")
        else:
            self.state = {
                'total_games': 0,
                'total_positions': 0,
                'players_processed': set(),
                'collection_rounds': 0,
                'last_update': None
            }
            logger.info("🆕 Starting fresh collection")
    
    def save_state(self):
        """Save collection state"""
        data = {
            'total_games': self.state['total_games'],
            'total_positions': self.state['total_positions'],
            'players_processed': list(self.state['players_processed']),
            'collection_rounds': self.state['collection_rounds'],
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def make_request(self, url: str, retries: int = 3):
        """Make HTTP request with retries"""
        self.rate_limit()
        
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"⚠️  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    return None
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None
    
    def get_leaderboard_players(self, category: str, top_n: int = 500) -> List[str]:
        """Get top players from leaderboard"""
        data = self.make_request("https://api.chess.com/pub/leaderboards")
        if not data or category not in data:
            return []
        
        players = []
        for player in data[category][:top_n]:
            username = player.get('username')
            if username:
                players.append(username.lower())
        
        return players
    
    def get_player_archives(self, username: str) -> List[str]:
        """Get monthly archives for player"""
        data = self.make_request(f"https://api.chess.com/pub/player/{username}/games/archives")
        return data.get('archives', []) if data else []
    
    def get_archive_games(self, archive_url: str) -> List[dict]:
        """Get games from archive"""
        data = self.make_request(archive_url)
        return data.get('games', []) if data else []
    
    def estimate_positions(self, game: dict) -> int:
        """Estimate positions in game"""
        pgn = game.get('pgn', '')
        moves = pgn.count('. ')
        return max(moves, 1)
    
    def collect_player(self, username: str) -> tuple:
        """Collect all games from player"""
        archives = self.get_player_archives(username)
        if not archives:
            return 0, 0
        
        total_games = 0
        total_positions = 0
        
        # Get last 12 months
        for archive_url in reversed(archives[-12:]):
            games = self.get_archive_games(archive_url)
            
            if games:
                # Save to dataset
                entry = {
                    'player': username,
                    'archive': archive_url,
                    'collected_at': datetime.now().isoformat(),
                    'games': games
                }
                
                with open(self.dataset_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
                
                # Count
                for game in games:
                    total_positions += self.estimate_positions(game)
                total_games += len(games)
        
        return total_games, total_positions
    
    def collect_category(self, category: str, top_n: int = 500):
        """Collect from category"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 CATEGORY: {category.upper()} (Top {top_n})")
        logger.info(f"{'='*80}\n")
        
        players = self.get_leaderboard_players(category, top_n)
        logger.info(f"📥 Found {len(players)} players")
        
        new_players = 0
        category_games = 0
        category_positions = 0
        
        for i, username in enumerate(players, 1):
            # Skip if already processed
            if username in self.state['players_processed']:
                continue
            
            logger.info(f"[{i}/{len(players)}] {username}...")
            
            games, positions = self.collect_player(username)
            
            if games > 0:
                self.state['players_processed'].add(username)
                self.state['total_games'] += games
                self.state['total_positions'] += positions
                category_games += games
                category_positions += positions
                new_players += 1
                
                logger.info(f"  ✅ {games} games, ~{positions:,} positions")
                
                # Save every 10 players
                if new_players % 10 == 0:
                    self.save_state()
        
        logger.info(f"\n✅ {category}: +{new_players} players, +{category_games:,} games, +{category_positions:,} positions")
        logger.info(f"📊 TOTAL: {self.state['total_games']:,} games, {self.state['total_positions']:,} positions\n")
        
        self.save_state()
    
    def run_forever(self):
        """Run continuous collection"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 MEGA CHESS.COM COLLECTOR")
        logger.info(f"{'='*80}")
        logger.info(f"🎯 GOAL: 100,000,000+ POSITIONS")
        logger.info(f"📊 Current: {self.state['total_positions']:,} positions ({(self.state['total_positions']/100_000_000)*100:.3f}%)")
        logger.info(f"{'='*80}\n")
        
        categories = [
            'live_bullet',
            'live_blitz', 
            'live_rapid',
            'daily',
            'daily960'
        ]
        
        try:
            while True:
                self.state['collection_rounds'] += 1
                round_num = self.state['collection_rounds']
                
                logger.info(f"\n{'#'*80}")
                logger.info(f"🔄 ROUND #{round_num}")
                logger.info(f"{'#'*80}\n")
                
                for category in categories:
                    self.collect_category(category, top_n=500)
                    time.sleep(10)  # Pause between categories
                
                # Round complete
                progress = (self.state['total_positions'] / 100_000_000) * 100
                logger.info(f"\n{'#'*80}")
                logger.info(f"✅ ROUND #{round_num} COMPLETE")
                logger.info(f"{'#'*80}")
                logger.info(f"📊 Games: {self.state['total_games']:,}")
                logger.info(f"📊 Positions: {self.state['total_positions']:,}")
                logger.info(f"📊 Players: {len(self.state['players_processed']):,}")
                logger.info(f"📊 Progress: {progress:.3f}% to 100M")
                logger.info(f"{'#'*80}\n")
                
                if self.state['total_positions'] >= 100_000_000:
                    logger.info(f"🎉🎉🎉 100M POSITIONS REACHED! 🎉🎉🎉")
                    logger.info(f"🔄 Continuing to expand dataset...\n")
                
                time.sleep(30)  # Pause before next round
        
        except KeyboardInterrupt:
            logger.info(f"\n⏹️  Stopped")
            logger.info(f"📊 Final: {self.state['total_games']:,} games, {self.state['total_positions']:,} positions")
            self.save_state()

def main():
    collector = MegaChessComCollector()
    collector.run_forever()

if __name__ == '__main__':
    main()
