#!/usr/bin/env python3

"""
Anti-Stockfish Process 1: Ultimate Chess.com Collector
GOAL: 100,000,000+ games and positions
Strategy:
1. Fetch ALL GMs (Grandmasters)
2. Sort them by Rapid Rating (Highest to Lowest)
3. Take Top 1000 from this sorted list
4. Top 100: Sync ALL new games
5. Top 101-1000: Dig 1000 games deeper into history per run
6. Loop forever
"""

import requests
import time
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

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
    def __init__(self):
        self.data_dir = Path("neural_network/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_file = self.data_dir / "chesscom_master_dataset.jsonl"
        self.state_file = Path("process1_state.json")
        self.gm_cache_file = Path("gm_ratings_cache.json")
        
        self.REQUEST_DELAY = 0.25
        self.last_request_time = 0
        self.headers = {
            'User-Agent': 'Anti-Stockfish-Collector/1.0 (contact: anti-stockfish@example.com)'
        }
        
        self.load_state()
    
    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    self.state = {
                        'total_games': data.get('total_games', 0),
                        'total_positions': data.get('total_positions', 0),
                        'collection_rounds': data.get('collection_rounds', 0),
                        'top_100_last_sync': data.get('top_100_last_sync', {}),
                        'dig_state': data.get('dig_state', {})
                    }
                logger.info(f"📊 Loaded State: {self.state['total_games']:,} games collected")
            except Exception as e:
                logger.error(f"❌ Error loading state: {e}")
                self.reset_state()
        else:
            self.reset_state()
            
    def reset_state(self):
        self.state = {
            'total_games': 0,
            'total_positions': 0,
            'collection_rounds': 0,
            'top_100_last_sync': {},
            'dig_state': {}
        }
        logger.info("🆕 Starting fresh collection state")

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def make_request(self, url: str, retries: int = 3):
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
                time.sleep(2 ** attempt)
        return None

    def get_player_stats(self, username: str) -> dict:
        """Get player stats including ratings"""
        return self.make_request(f"https://api.chess.com/pub/player/{username}/stats")

    def get_sorted_gms(self) -> List[str]:
        """Get ALL GMs, fetch their Rapid ratings, and sort them. Uses cache if available."""
        
        # Check cache first
        if self.gm_cache_file.exists():
            try:
                with open(self.gm_cache_file) as f:
                    cache_data = json.load(f)
                    cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01T00:00:00'))
                    
                    # Use cache FOREVER as requested
                    logger.info(f"📂 Loaded sorted GM list from cache ({len(cache_data['players'])} players, created {cache_time})")
                    return cache_data['players']
            except Exception as e:
                logger.warning(f"⚠️  Failed to load GM cache: {e}")

        logger.info("📥 Fetching list of all GMs...")
        data = self.make_request("https://api.chess.com/pub/titled/GM")
        if not data or 'players' not in data:
            return []
        
        all_gms = data['players']
        logger.info(f"📋 Found {len(all_gms)} GMs. Fetching ratings to sort...")
        
        gm_ratings = []
        count = 0
        
        for username in all_gms:
            stats = self.get_player_stats(username)
            rating = 0
            if stats and 'chess_rapid' in stats:
                rating = stats['chess_rapid'].get('last', {}).get('rating', 0)
            
            gm_ratings.append({'username': username, 'rating': rating})
            count += 1
            if count % 100 == 0:
                logger.info(f"  ...fetched ratings for {count}/{len(all_gms)} GMs")
                
        # Sort by rating descending
        gm_ratings.sort(key=lambda x: x['rating'], reverse=True)
        
        sorted_usernames = [p['username'] for p in gm_ratings]
        
        # Save to cache
        try:
            with open(self.gm_cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'players': sorted_usernames,
                    'ratings': gm_ratings # Save ratings too for debugging/verification
                }, f, indent=2)
            logger.info(f"💾 Saved sorted GM list to {self.gm_cache_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save GM cache: {e}")
        
        # Log top 10 for verification
        logger.info("🏆 Top 10 GMs by Rapid Rating:")
        for i, p in enumerate(gm_ratings[:10]):
            logger.info(f"  #{i+1} {p['username']} ({p['rating']})")
            
        return sorted_usernames

    def get_archives(self, username: str) -> List[str]:
        data = self.make_request(f"https://api.chess.com/pub/player/{username}/games/archives")
        return data.get('archives', []) if data else []

    def get_games(self, url: str) -> List[dict]:
        data = self.make_request(url)
        return data.get('games', []) if data else []

    def estimate_positions(self, game: dict) -> int:
        return max(game.get('pgn', '').count('. '), 1)

    def filter_game(self, game: dict) -> bool:
        """Strict filter for high quality games"""
        return (
            game.get('rules') == 'chess' and 
            game.get('time_class') in ['rapid', 'classical', 'blitz', 'daily']
        )

    def sync_top_player(self, username: str):
        """Top 100 Strategy: Get ALL new games since last sync"""
        last_sync = self.state['top_100_last_sync'].get(username, 0)
        archives = self.get_archives(username)
        
        # If last_sync is 0 (first run), we want ALL archives.
        # If last_sync > 0 (subsequent runs), we only need recent archives.
        if last_sync == 0:
            target_archives = archives 
            logger.info(f"  🆕 First run for {username}: Fetching ALL archives...")
        else:
            target_archives = archives[-3:] # Check last 3 months for updates
            
        new_games_count = 0
        new_positions_count = 0
        max_timestamp = last_sync
        
        for url in target_archives:
            games = self.get_games(url)
            for game in games:
                if not self.filter_game(game):
                    continue
                    
                end_time = game.get('end_time', 0)
                if end_time > last_sync:
                    self.save_game(username, url, game)
                    new_games_count += 1
                    new_positions_count += self.estimate_positions(game)
                    max_timestamp = max(max_timestamp, end_time)
        
        self.state['top_100_last_sync'][username] = max_timestamp
        return new_games_count, new_positions_count

    def dig_deep_player(self, username: str, limit: int = 1000):
        """101-1000 Strategy: Dig deeper into history (chunk of 1000)"""
        archives = self.get_archives(username)
        if not archives:
            return 0, 0
            
        archives = list(reversed(archives))
        
        dig_state = self.state['dig_state'].get(username, {'archive_idx': 0, 'game_offset': 0})
        current_idx = dig_state['archive_idx']
        current_offset = dig_state['game_offset']
        
        collected_count = 0
        collected_positions = 0
        
        while collected_count < limit and current_idx < len(archives):
            url = archives[current_idx]
            games = self.get_games(url)
            games = list(reversed(games))
            
            if current_offset < len(games):
                games_to_process = games[current_offset:]
                
                for game in games_to_process:
                    if collected_count >= limit:
                        break
                        
                    if self.filter_game(game):
                        self.save_game(username, url, game)
                        collected_count += 1
                        collected_positions += self.estimate_positions(game)
                    
                    current_offset += 1
            
            if current_offset >= len(games):
                current_idx += 1
                current_offset = 0
        
        self.state['dig_state'][username] = {
            'archive_idx': current_idx,
            'game_offset': current_offset
        }
        
        return collected_count, collected_positions

    def save_game(self, username, archive_url, game):
        entry = {
            'player': username,
            'archive': archive_url,
            'collected_at': datetime.now().isoformat(),
            'games': [game]
        }
        with open(self.dataset_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
            
        self.state['total_games'] += 1
        self.state['total_positions'] += self.estimate_positions(game)

    def run_forever(self):
        logger.info("🚀 STARTING TOP 1000 GM COLLECTOR (SORTED BY RATING)")
        
        while True:
            self.state['collection_rounds'] += 1
            logger.info(f"\n🔄 ROUND {self.state['collection_rounds']} START")
            
            # 1. Get Sorted GMs
            # We do this every round to keep ratings fresh
            players = self.get_sorted_gms()
            
            if not players:
                logger.error("❌ Failed to get GMs, sleeping...")
                time.sleep(60)
                continue
                
            # Take Top 1000
            top_1000 = players[:1000]
            logger.info(f"📋 Processing Top {len(top_1000)} GMs")
            
            # 2. Process Top 100 (Sync All)
            logger.info("👑 Processing TOP 100 (Syncing ALL new games)...")
            for i, p in enumerate(top_1000[:100]):
                g, pos = self.sync_top_player(p)
                if g > 0:
                    logger.info(f"  [{i+1}/100] {p}: +{g} games")
                self.save_state()  # Save after EVERY player to allow safe restarts
            
            # 3. Process 101-1000 (Dig 1000)
            logger.info("⛏️  Processing 101-1000 (Digging 1000 games each)...")
            for i, p in enumerate(top_1000[100:]):
                real_rank = i + 101
                g, pos = self.dig_deep_player(p, limit=1000)
                if g > 0:
                    logger.info(f"  [{real_rank}/1000] {p}: +{g} games (Deep Dig)")
                self.save_state()  # Save after EVERY player to allow safe restarts
            
            self.save_state()
            logger.info("✅ Round Complete. Sleeping 60s...")
            time.sleep(60)

if __name__ == '__main__':
    MegaChessComCollector().run_forever()
