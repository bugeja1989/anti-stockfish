import requests
import zipfile
import io
import chess.pgn
import json
from pathlib import Path
import urllib3

# Suppress InsecureRequestWarning since we are disabling SSL verify
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# URLs for Attacking Legends
DATA_SOURCES = {
    "Tal": "https://www.pgnmentor.com/players/Tal.zip",
    "Kasparov": "https://www.pgnmentor.com/players/Kasparov.zip",
    "Nakamura": "https://www.pgnmentor.com/players/Nakamura.zip",
    "Carlsen": "https://www.pgnmentor.com/players/Carlsen.zip"
}

DATA_DIR = Path("neural_network/data")
OUTPUT_FILE = DATA_DIR / "extracted_positions.jsonl"

def download_and_extract(name, url):
    print(f"⬇️ Downloading games of {name}...")
    try:
        # Add headers to mimic a browser and avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        # Disable SSL verification to bypass LibreSSL issues on some Macs
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(response.content))
        
        # Extract PGN file
        pgn_filename = z.namelist()[0]
        print(f"   Extracted {pgn_filename}")
        return z.read(pgn_filename).decode('latin-1') # PGNs often use latin-1
    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")
        return None

def process_pgn(pgn_content, label_value=1.0, chaos_value=0.9):
    print("   Processing games...")
    pgn_io = io.StringIO(pgn_content)
    new_positions = []
    game_count = 0
    
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
            
        board = game.board()
        result = game.headers.get("Result", "*")
        
        # Only learn from WINS (1-0 or 0-1)
        # We want to learn HOW they won.
        if result == "1-0":
            winner = chess.WHITE
        elif result == "0-1":
            winner = chess.BLACK
        else:
            continue # Skip draws
            
        game_count += 1
        
        for move in game.mainline_moves():
            fen = board.fen()
            
            # We only learn from the WINNER's moves
            if board.turn == winner:
                entry = {
                    "fen": fen,
                    "move": move.uci(),
                    "outcome": label_value, # 1.0 = Good Move
                    "chaos": chaos_value    # 0.9 = High Chaos/Aggression (Tal Style)
                }
                new_positions.append(entry)
            
            board.push(move)
            
    print(f"   Processed {game_count} winning games.")
    return new_positions

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_positions = []
    
    for name, url in DATA_SOURCES.items():
        pgn_content = download_and_extract(name, url)
        if pgn_content:
            positions = process_pgn(pgn_content)
            all_positions.extend(positions)
            print(f"   Added {len(positions)} positions from {name}")

    if not all_positions:
        print("No positions extracted.")
        return

    print(f"💾 Appending {len(all_positions)} positions to dataset...")
    
    with open(OUTPUT_FILE, "a") as f:
        for pos in all_positions:
            f.write(json.dumps(pos) + "\n")
            
    print("✅ Done! The engine will now learn from the masters of attack.")

if __name__ == "__main__":
    main()
