# xử lý & load dữ liệu

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class CaroDataset(Dataset):
    # Dataset với ĐẦY ĐỦ labels cho cả value head VÀ policy head
    def __init__(self, data_dir="data/selfplay", min_game_length=20):
        self.boards = []
        self.results = []
        self.move_indices = []
        self.min_game_length = min_game_length
        self._load_data(data_dir)
        
    def _load_data(self, data_dir):
        # Load và xử lý dữ liệu từ JSON files
        print(f"Loading data from {data_dir}...")
        
        loaded_games = 0
        skipped_games = 0
        
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".json"):
                continue
                
            with open(os.path.join(data_dir, fname), "r") as f:
                game = json.load(f)
            
            moves = game["moves"]
            result = game.get("result", game.get("winner", 0))

            
            # Bỏ games quá ngắn (không có giá trị học)
            if len(moves) < self.min_game_length:
                skipped_games += 1
                continue
            
            # Reconstruct all intermediate states
            board = np.zeros((15, 15), dtype=np.int8)
            player = 1
            
            for move_idx, (r, c) in enumerate(moves):
                # State TRƯỚC khi đánh
                player_layer = (board == player).astype(np.float32)
                opp_layer = (board == -player).astype(np.float32)
                state = np.stack([player_layer, opp_layer], axis=0)
                
                # Save state, result, và MOVE
                self.boards.append(state)
                self.results.append(result if player == 1 else -result)
                
                # Convert move (r,c) thành index cho policy head
                move_index = r * 15 + c  # Flatten 2D -> 1D
                self.move_indices.append(move_index)
                
                # Place stone
                board[r, c] = player
                player = -player
            
            loaded_games += 1
            
            if loaded_games % 100 == 0:
                print(f"  Loaded {loaded_games} games...", end='\r')
        
        # Convert to numpy arrays
        self.boards = np.array(self.boards, dtype=np.float32)
        self.results = np.array(self.results, dtype=np.float32)
        self.move_indices = np.array(self.move_indices, dtype=np.int64)
        
        print(f"\nLoaded {loaded_games} games ({skipped_games} skipped)")
        print(f" Total positions: {len(self.boards)}")
        print(f" Avg moves/game: {len(self.boards)/max(1, loaded_games):.1f}")
        
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        """ Return 3 values: board, result, move_index"""
        x = torch.tensor(self.boards[idx])
        y = torch.tensor(self.results[idx])
        m = torch.tensor(self.move_indices[idx], dtype=torch.long)
        
        return x, y, m


# DATA QUALITY ANALYSIS
def analyze_dataset(data_dir="data/selfplay"):
    """Phân tích chất lượng dataset"""
    print("\n" + "="*60)
    print("🔍 DATASET QUALITY ANALYSIS")
    print("="*60)
    
    game_lengths = []
    depths = []
    strategies = []
    results = {'win': 0, 'loss': 0, 'draw': 0}
    
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
            
        with open(os.path.join(data_dir, fname), "r") as f:
            game = json.load(f)
        
        game_lengths.append(len(game["moves"]))
        
        if "metadata" in game:
            meta = game["metadata"]
            depths.append(meta.get("max_depth", 0))
            strategies.append(meta.get("strategy", "unknown"))
        
        result = game.get("result", game.get("winner", 0))
        if result == 1:
            results['win'] += 1
        elif result == -1:
            results['loss'] += 1
        else:
            results['draw'] += 1
    
    print(f"Total games: {len(game_lengths)}")
    print(f"Avg game length: {np.mean(game_lengths):.1f} ± {np.std(game_lengths):.1f}")
    print(f"Min/Max length: {np.min(game_lengths)} / {np.max(game_lengths)}")
    
    if depths:
        print(f"\nAvg search depth: {np.mean(depths):.1f}")
        print(f"Depth distribution: {dict(zip(*np.unique(depths, return_counts=True)))}")
    
    if strategies:
        print(f"\nStrategy distribution: {dict(zip(*np.unique(strategies, return_counts=True)))}")
    
    print(f"\nResult distribution:")
    print(f"  Player 1 wins: {results['win']} ({results['win']/len(game_lengths)*100:.1f}%)")
    print(f"  Player 2 wins: {results['loss']} ({results['loss']/len(game_lengths)*100:.1f}%)")
    print(f"  Draws: {results['draw']} ({results['draw']/len(game_lengths)*100:.1f}%)")
    


if __name__ == "__main__":
    analyze_dataset("data/selfplay")
    dataset = CaroDataset("data/selfplay")
    print(f"Dataset loaded: {len(dataset)} positions")
