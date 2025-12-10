#!/usr/bin/env python3

"""
Anti-Stockfish Process 1: Ultimate Chess.com Collector
GOAL: 100,000,000+ games and positions
Strategy:
1. Prioritize Super GMs (Classical/Rapid)
2. Collect from ALL top charts (Top 100)
3. Filter for high-quality games (Classical/Rapid/Blitz)
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
        
        # Chess.com requires a User-Agent with contact info
        self.headers = {
            'User-Agent': 'Anti-Stockfish-Collector/1.0 (contact: anti-stockfish@example.com)'
        }
        
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
                response = requests.get(url, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"⚠️  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"⚠️  Request failed: {url} -> {response.status_code}")
                    return None
            except Exception as e:
                logger.error(f"❌ Request error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None
    
    def get_leaderboard_players(self, category: str, top_n: int = 100) -> List[str]:
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
    
    def get_titled_players(self, title: str) -> List[str]:
        """Get all players with a specific title (GM, IM, etc.)"""
        data = self.make_request(f"https://api.chess.com/pub/titled/{title}")
        return data.get('players', []) if data else []

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
        
        # Get last 24 months (2 years) for high quality games
        for archive_url in reversed(archives[-24:]):
            games = self.get_archive_games(archive_url)
            
            if games:
                # Filter for Classical/Rapid/Blitz (exclude Bullet/Hyperbullet if possible, but we take all for volume)
                # Prioritize standard chess
                filtered_games = [
                    g for g in games 
                    if g.get('rules') == 'chess' and g.get('time_class') in ['rapid', 'classical', 'blitz', 'daily']
                ]
                
                if not filtered_games:
                    continue

                # Save to dataset
                entry = {
                    'player': username,
                    'archive': archive_url,
                    'collected_at': datetime.now().isoformat(),
                    'games': filtered_games
                }
                
                with open(self.dataset_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
                
                # Count
                for game in filtered_games:
                    total_positions += self.estimate_positions(game)
                total_games += len(filtered_games)
        
        return total_games, total_positions
    
    def collect_category(self, category: str, top_n: int = 100):
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

    def collect_titled_players(self, title: str):
        """Collect all players with a specific title"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🏆 TITLE: {title} (All Players)")
        logger.info(f"{'='*80}\n")
        
        players = self.get_titled_players(title)
        logger.info(f"📥 Found {len(players)} {title}s")
        
        new_players = 0
        
        for i, username in enumerate(players, 1):
            if username in self.state['players_processed']:
                continue
                
            logger.info(f"[{i}/{len(players)}] {username} ({title})...")
            games, positions = self.collect_player(username)
            
            if games > 0:
                self.state['players_processed'].add(username)
                self.state['total_games'] += games
                self.state['total_positions'] += positions
                new_players += 1
                logger.info(f"  ✅ {games} games, ~{positions:,} positions")
                
                if new_players % 10 == 0:
                    self.save_state()
        
        self.save_state()
    
    def run_forever(self):
        """Run continuous collection"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 MEGA CHESS.COM COLLECTOR")
        logger.info(f"{'='*80}")
        logger.info(f"🎯 GOAL: 100,000,000+ POSITIONS")
        logger.info(f"📊 Current: {self.state['total_positions']:,} positions ({(self.state['total_positions']/100_000_000)*100:.3f}%)")
        logger.info(f"{'='*80}\n")
        
        # Priority Order:
        # 1. Daily (often corresponds to Classical/Correspondence quality)
        # 2. Live Rapid (High quality)
        # 3. Live Blitz (Good volume, decent quality)
        # 4. Live Bullet (High volume, lower quality - kept for tactical patterns)
        categories = [
            'daily',
            'live_rapid',
            'live_blitz',
            'live_bullet',
            'live_bughouse', # Sometimes fun patterns
            'live_blitz960',
            'daily960'
        ]
        
        try:
            while True:
                self.state['collection_rounds'] += 1
                round_num = self.state['collection_rounds']
                
                logger.info(f"\n{'#'*80}")
                logger.info(f"🔄 ROUND #{round_num}")
                logger.info(f"{'#'*80}\n")
                
                # 1. First Priority: Super GMs (GM Title)
                # We do this every few rounds to catch new games from GMs
                if round_num % 5 == 1:
                    self.collect_titled_players('GM')
                
                # 2. Top Charts (Top 100 of each category)
                for category in categories:
                    self.collect_category(category, top_n=100)
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
