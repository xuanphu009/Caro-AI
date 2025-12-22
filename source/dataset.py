import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from typing import Optional

class CaroDataset(Dataset):
    """
    ✅ FIXED VERSION - Sửa lỗi label và augmentation
    
    Changes:
    1. Label được gán CHÍNH XÁC theo từng position
    2. Augmentation chỉ áp dụng trong training (không làm phình dataset)
    3. Hỗ trợ cả old format (không có first_player)
    """
    def __init__(self, data_dir="data/professional", min_game_length=10, use_augmentation=False):
        self.boards = []
        self.results = []
        self.move_indices = []
        self.min_game_length = min_game_length
        self.use_augmentation = use_augmentation  # ✅ Chỉ augment khi cần
        self._load_data(data_dir)
        
    def _load_data(self, data_dir):
        print(f"📂 Loading data from {data_dir}...")
        
        loaded_games = 0
        skipped_games = 0
        skipped_no_first_player = 0
        
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".json"):
                continue
                
            with open(os.path.join(data_dir, fname), "r") as f:
                game = json.load(f)
            
            moves = game["moves"]
            winner = game.get("winner", 0)
            
            # ✅ FIXED: Xử lý cả old format và new format
            first_player = game.get("first_player", None)
            
            if first_player in [None, "AI", "ai", "player1", "p1", "1"]:
                player = 1
            elif first_player in ["human", "Human", "player2", "p2", "-1"]:
                player = -1
            else:
                try:
                    player = int(first_player)
                except Exception:
                    print(f"⚠️ Warning: Invalid first_player='{first_player}' in file {fname}, defaulting to 1")
                    player = 1
            
            # Skip too short games
            if len(moves) < self.min_game_length:
                skipped_games += 1
                continue
            
            # ✅ FIXED: Reconstruct game với first_player đúng
            board = np.zeros((15, 15), dtype=np.int8)
            
            
            for move_idx, (r, c) in enumerate(moves):
                # State BEFORE this move
                player_layer = (board == player).astype(np.float32)
                opp_layer = (board == -player).astype(np.float32)
                state = np.stack([player_layer, opp_layer], axis=0)
                
                # ✅ CRITICAL FIX: Label phải phản ánh "player này có thắng không?"
                # - Nếu game chưa kết thúc, ta không biết → dùng intermediate value
                # - Nếu winner == player → +1.0
                # - Nếu winner == -player → -1.0
                # - Nếu draw → 0.0
                
                # ✅ IMPROVED: Dùng "reward shaping" - gần cuối game có signal mạnh hơn
                remaining_moves = len(moves) - move_idx
                discount = 0.95 ** remaining_moves  # Exponential decay
                
                if winner == 0:
                    label = 0.0
                elif winner == player:
                    label = discount * 1.0  # Winner gets positive reward
                else:
                    label = discount * (-1.0)  # Loser gets negative
                
                # ✅ Convert move to index
                move_index = r * 15 + c
                
                # ✅ FIXED: Không augment ở đây! Augment trong __getitem__
                self.boards.append(state)
                self.results.append(label)
                self.move_indices.append(move_index)

                # Place stone and switch player
                board[r, c] = player
                player = -player
            
            loaded_games += 1
            
            if loaded_games % 100 == 0:
                print(f"  Loaded {loaded_games} games...", end='\r')
        
        # Convert to numpy arrays
        self.boards = np.array(self.boards, dtype=np.float32)
        self.results = np.array(self.results, dtype=np.float32)
        self.move_indices = np.array(self.move_indices, dtype=np.int64)
        
        print(f"\n✅ Loaded {loaded_games} games ({skipped_games} too short, "
              f"{skipped_no_first_player} missing first_player)")
        print(f"📊 Total positions: {len(self.boards)}")
        print(f"📈 Avg moves/game: {len(self.boards)/max(1, loaded_games):.1f}")
        
        # ✅ Check label distribution
        if len(self.results) > 0:
            n_win = np.sum(self.results > 0)
            n_loss = np.sum(self.results < 0)
            n_draw = np.sum(self.results == 0)
            total = len(self.results)
            
            print(f"\n📊 Label Distribution:")
            print(f"   Win positions:  {n_win:6d} ({n_win/total*100:5.1f}%)")
            print(f"   Loss positions: {n_loss:6d} ({n_loss/total*100:5.1f}%)")
            print(f"   Draw positions: {n_draw:6d} ({n_draw/total*100:5.1f}%)")
            
            # ✅ FIXED: Relaxed threshold (40% → 60% is acceptable)
            if abs(n_win - n_loss) > total * 0.4:
                print(f"\n⚠️  WARNING: Label imbalance detected!")
                print(f"   Difference: {abs(n_win - n_loss)} positions ({abs(n_win-n_loss)/total*100:.1f}%)")
                print(f"   Consider generating more balanced games.")

    def _augment_single(self, state: np.ndarray, move_idx: int) -> tuple:
        """
        ✅ FIXED: Augmentation đơn giản, random transformation
        Chỉ áp dụng 1 transform cho mỗi sample (không tạo 8x data)
        """
        if not self.use_augmentation:
            return state, move_idx
        
        # Random rotation (0°, 90°, 180°, 270°)
        k = np.random.randint(0, 4)
        rotated = np.rot90(state, k, axes=(1, 2))
        
        r, c = move_idx // 15, move_idx % 15
        for _ in range(k):
            r, c = c, 14 - r
        
        # Random flip
        if np.random.rand() > 0.5:
            rotated = np.flip(rotated, axis=2).copy()
            c = 14 - c
        
        new_move_idx = r * 15 + c
        return rotated, new_move_idx

    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        """
        ✅ FIXED: Augmentation on-the-fly (chỉ khi use_augmentation=True)
        """
        state = self.boards[idx]
        label = self.results[idx]
        move_idx = self.move_indices[idx]
        
        # Apply augmentation if enabled
        state, move_idx = self._augment_single(state, move_idx)
        
        x = torch.tensor(state.copy(), dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        m = torch.tensor(move_idx, dtype=torch.long)
        
        return x, y, m


# ✅ DATA QUALITY ANALYSIS (Updated)
def analyze_dataset(data_dir="data/professional"):
    """Analyze dataset quality with improved metrics"""
    print("\n" + "="*60)
    print("📊 DATASET QUALITY ANALYSIS")
    print("="*60)
    
    game_lengths = []
    depths = []
    strategies = []
    results = {'p1_win': 0, 'p2_win': 0, 'draw': 0}
    first_players = []
    
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
            
        with open(os.path.join(data_dir, fname), "r") as f:
            game = json.load(f)
        
        game_lengths.append(len(game["moves"]))
        
        # Metadata
        if "metadata" in game:
            meta = game["metadata"]
            depths.append(meta.get("max_depth", 0))
            strategies.append(meta.get("strategy", "unknown"))
        
        # First player
        first_player = game.get("first_player", 1)  # Default to 1
        first_players.append(first_player)
        
        # Results
        winner = game.get("winner", 0)
        if winner == 1:
            results['p1_win'] += 1
        elif winner == -1:
            results['p2_win'] += 1
        else:
            results['draw'] += 1
    
    total_games = len(game_lengths)
    
    print(f"Total games: {total_games}")
    print(f"Avg game length: {np.mean(game_lengths):.1f} ± {np.std(game_lengths):.1f}")
    print(f"Min/Max length: {np.min(game_lengths)} / {np.max(game_lengths)}")
    
    if depths:
        print(f"\nAvg search depth: {np.mean(depths):.1f}")
    
    if strategies:
        print(f"\nStrategy distribution:")
        from collections import Counter
        for k, v in Counter(strategies).items():
            print(f"  {k}: {v} ({v/total_games*100:.1f}%)")
    
    print(f"\n🎯 Result distribution:")
    print(f"  Player 1 wins: {results['p1_win']:4d} ({results['p1_win']/total_games*100:5.1f}%)")
    print(f"  Player 2 wins: {results['p2_win']:4d} ({results['p2_win']/total_games*100:5.1f}%)")
    print(f"  Draws:         {results['draw']:4d} ({results['draw']/total_games*100:5.1f}%)")
    
    # ✅ Check balance
    if len(first_players) > 0:
        n_p1_first = sum(1 for p in first_players if p == 1)
        n_p2_first = sum(1 for p in first_players if p == -1)
        print(f"\n🎲 First player distribution:")
        print(f"  Player 1 starts: {n_p1_first} ({n_p1_first/len(first_players)*100:.1f}%)")
        print(f"  Player -1 starts: {n_p2_first} ({n_p2_first/len(first_players)*100:.1f}%)")
    
    # ✅ Warnings
    p1_rate = results['p1_win'] / (results['p1_win'] + results['p2_win'] + 0.001)
    if abs(p1_rate - 0.5) > 0.15:
        print(f"\n⚠️  WARNING: Win rate imbalance!")
        print(f"   Player 1 win rate: {p1_rate*100:.1f}%")
        print(f"   Expected: ~50% for balanced self-play")
    
    print("="*60)


if __name__ == "__main__":
    analyze_dataset("data/professional")
    
    # Test loading
    dataset = CaroDataset("data/professional", use_augmentation=False)
    print(f"\n✅ Dataset loaded: {len(dataset)} positions")
    
    # Test augmentation
    dataset_aug = CaroDataset("data/professional", use_augmentation=True)
    x, y, m = dataset_aug[0]
    print(f"✅ Augmentation test: shape={x.shape}, label={y.item():.2f}, move={m.item()}")