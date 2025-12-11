import chess
import chess.pgn
import json
import io
import requests
from pathlib import Path

def download_file(url, target_path):
    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(target_path, 'wb') as f:
        f.write(response.content)
    print(f"Saved to {target_path}")

def inject_opening_book():
    # Base URL for hayatbiralem/eco.json repository
    base_url = "https://raw.githubusercontent.com/hayatbiralem/eco.json/master"
    files = ['ecoA.json', 'ecoB.json', 'ecoC.json', 'ecoD.json', 'ecoE.json']
    
    # Output paths
    data_dir = Path("neural_network/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / "extracted_positions.jsonl"
    
    # Load existing FENs to avoid duplicates
    existing_fens = set()
    if output_path.exists():
        print(f"Scanning existing data in {output_path}...")
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        existing_fens.add(data['fen'])
                    except:
                        continue
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")
            
    print(f"Found {len(existing_fens)} existing positions.")
    
    positions_added = 0
    
    for filename in files:
        url = f"{base_url}/{filename}"
        local_path = data_dir / filename
        
        # Download if not exists
        try:
            download_file(url, local_path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            continue
            
        print(f"Processing {filename}...")
        
        try:
            with open(local_path, 'r') as f:
                openings = json.load(f)
                
            print(f"Loaded {len(openings)} openings from {filename}.")
            
            with open(output_path, 'a') as out_f:
                # The JSON is a dictionary where keys are FENs and values are opening objects
                # We iterate over the values
                for fen_key, op in openings.items():
                    eco_code = op.get('eco', '')
                    name = op.get('name', '')
                    # The JSON has 'moves' as a string like "1. e4 e5 2. Nf3..."
                    pgn_moves = op.get('moves', '')
                    
                    if not pgn_moves:
                        continue
                        
                    try:
                        pgn = io.StringIO(pgn_moves)
                        game = chess.pgn.read_game(pgn)
                        
                        if not game:
                            continue
                            
                        board = game.board()
                        move_number = 1
                        
                        for move in game.mainline_moves():
                            fen = board.fen()
                            
                            # Skip if already in dataset
                            if fen in existing_fens:
                                board.push(move)
                                move_number += 1
                                continue
                            
                            # Create training entry
                            # Label: 1.0 (Win/Best Move)
                            # We want the model to treat opening book moves as "perfect" play.
                            position = {
                                'fen': fen,
                                'move': move.uci(),
                                'outcome': 1.0, # Treat book moves as winning moves
                                'move_number': move_number,
                                'source': 'opening_book_eco_12k',
                                'opening_name': name,
                                'eco': eco_code
                            }
                            
                            out_f.write(json.dumps(position) + '\n')
                            existing_fens.add(fen)
                            positions_added += 1
                            
                            board.push(move)
                            move_number += 1
                            
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
                
    print(f"Successfully injected {positions_added} NEW opening book positions.")
    print(f"Total positions in dataset: {len(existing_fens)}")

if __name__ == "__main__":
    inject_opening_book()
