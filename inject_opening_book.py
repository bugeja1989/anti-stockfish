import chess
import chess.pgn
import json
import io
from pathlib import Path

def inject_opening_book():
    # Input/Output paths
    tsv_path = Path("/home/ubuntu/page_texts/raw.githubusercontent.com_niklasf_eco_master_a.tsv.md")
    output_path = Path("/home/ubuntu/anti-stockfish/neural_network/data/extracted_positions.jsonl")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading opening book from {tsv_path}...")
    
    positions_added = 0
    
    with open(tsv_path, 'r') as f:
        lines = f.readlines()
        
    # Skip header lines (first 6 lines based on file inspection)
    start_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("eco\tname\tpgn"):
            start_line = i + 1
            break
            
    print(f"Starting processing from line {start_line}...")
    
    with open(output_path, 'a') as out_f:
        for line in lines[start_line:]:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
                
            eco_code = parts[0]
            name = parts[1]
            pgn_moves = parts[2]
            
            # Parse PGN moves
            try:
                pgn = io.StringIO(pgn_moves)
                game = chess.pgn.read_game(pgn)
                
                if not game:
                    continue
                    
                board = game.board()
                move_number = 1
                
                for move in game.mainline_moves():
                    fen = board.fen()
                    
                    # Create training entry
                    # We treat opening book moves as "good" moves (outcome 0.5 for drawish/solid, or slightly positive)
                    # Since these are established theory, we can label them as neutral/solid (0.5) 
                    # or slightly positive (0.6) to encourage the engine to follow them.
                    # Let's use 0.6 to give a slight preference over random moves.
                    
                    position = {
                        'fen': fen,
                        'move': move.uci(),
                        'outcome': 0.6,  # Slightly positive to encourage following book
                        'move_number': move_number,
                        'source': 'opening_book_eco',
                        'opening_name': name,
                        'eco': eco_code
                    }
                    
                    out_f.write(json.dumps(position) + '\n')
                    positions_added += 1
                    
                    board.push(move)
                    move_number += 1
                    
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {e}")
                continue
                
    print(f"Successfully injected {positions_added} opening book positions into {output_path}")

if __name__ == "__main__":
    inject_opening_book()
