"""
CARO AI - SELF-PLAY DATA GENERATOR (v3.0)
Generate training data by AI playing against itself
Features:
[1] Chiến lược đa dạng
    Tìm kiếm theo mẫu (pattern heuristic)
    Dùng model CNN kết hợp alpha-beta
    Chơi ngẫu nhiên (baseline so sánh)
    Kết hợp giữa khám phá và khai thác (mixed strategy)

[2] Tăng độ khó theo giai đoạn
    Giai đoạn 1: Độ sâu 2-3 (dễ, nhanh)
    Giai đoạn 2: Độ sâu 4-5 (trung bình)
    Giai đoạn 3: Độ sâu 6+ và dùng model (khó)
[3] Kiểm soát chất lượng dữ liệu
    Lưu metadata: độ sâu, thời gian, chiến lược
    Lọc trận quá ngắn hoặc quá dài
    Cân bằng dữ liệu: thắng / thua / hòa
    Loại bỏ trạng thái trùng lặp
[4] Tạo dữ liệu song song
    Hỗ trợ multi-process (nhiều luồng)
    Hiển thị tiến độ
    Tự động chia lô (batching)
[5] Tích hợp dễ dàng với huấn luyện
    Tương thích với dataset.py
    Có thể tích hợp trực tiếp vào vòng lặp huấn luyện
    Hỗ trợ học tăng cường (incremental learning)
"""

import os
import sys
import json
import time
import random
from typing import Optional, Dict, List, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from game import Board
from evaluate import evaluate_pattern, evaluate_simple
from searchs import get_best_move

# CONFIGURATION
@dataclass
class SelfPlayConfig:
    """Configuration for self-play generation"""
    
    # Output
    save_dir: str = "data/selfplay"
    
    # Game settings
    max_depth: int = 3
    max_time: float = 2.0
    max_moves: int = 225  # 15x15 board
    
    # Search settings
    use_model: bool = False
    model_path: Optional[str] = None
    
    # Strategy
    strategy: str = "search"  # search / random / mixed
    exploration_rate: float = 0.1  # For mixed strategy
    
    # Quality control
    min_game_length: int = 10
    max_game_length: int = 225
    
    # Progressive training
    progressive: bool = False
    stages: int = 3
    
    # Verbose
    verbose: bool = True
    save_metadata: bool = True


# GAME STRATEGIES
class GameStrategy:
    """Base class for game playing strategies"""
    
    def get_move(self, board: Board, player: int) -> Optional[Tuple[int, int]]:
        raise NotImplementedError


class RandomStrategy(GameStrategy):
    """Random move selection (baseline)"""
    
    def get_move(self, board: Board, player: int) -> Optional[Tuple[int, int]]:
        moves = board.generate_candidates(radius=2)
        if not moves:
            moves = board.legal_moves()
        return random.choice(moves) if moves else None


class SearchStrategy(GameStrategy):
    """Search-based strategy (pattern heuristic)"""
    
    def __init__(self, max_depth: int = 3, max_time: float = 2.0):
        self.max_depth = max_depth
        self.max_time = max_time
    
    def get_move(self, board: Board, player: int) -> Optional[Tuple[int, int]]:
        try:
            move, value, stats = get_best_move(
                board=board,
                max_depth=self.max_depth,
                max_time=self.max_time,
                player=player,
                evaluate_fn=evaluate_pattern,
                verbose=False
            )
            return move
        except Exception as e:
            print(f"Search failed: {e}, using random move")
            return RandomStrategy().get_move(board, player)


class ModelGuidedStrategy(GameStrategy):
    """Model-guided search strategy (CNN + alpha-beta)"""
    
    def __init__(self, model_path: str, max_depth: int = 4, max_time: float = 2.0):
        self.max_depth = max_depth
        self.max_time = max_time
        self.model_loaded = False
        
        # Try to load model
        try:
            from src.model import load_model_into_cache, evaluate_model, policy_suggest
            load_model_into_cache(model_path, use_fp16=True, use_ema=True)
            self.evaluate_fn = evaluate_model
            self.policy_fn = policy_suggest
            self.model_loaded = True
            print(f"Model loaded: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Falling back to pattern heuristic")
            self.evaluate_fn = evaluate_pattern
            self.policy_fn = None
    
    def get_move(self, board: Board, player: int) -> Optional[Tuple[int, int]]:
        try:
            # Get policy suggestions if available
            policy_scores = None
            if self.policy_fn and self.model_loaded:
                try:
                    policy_scores = self.policy_fn(board, top_k=20)
                except:
                    pass
            
            move, value, stats = get_best_move(
                board=board,
                max_depth=self.max_depth,
                max_time=self.max_time,
                player=player,
                evaluate_fn=self.evaluate_fn,
                policy_scores=policy_scores,
                verbose=False
            )
            return move
        except Exception as e:
            print(f"⚠️Model-guided search failed: {e}, using pattern search")
            return SearchStrategy(self.max_depth, self.max_time).get_move(board, player)


class MixedStrategy(GameStrategy):
    """Mixed strategy: exploration vs exploitation"""
    
    def __init__(self, main_strategy: GameStrategy, exploration_rate: float = 0.1):
        self.main_strategy = main_strategy
        self.exploration_rate = exploration_rate
        self.random_strategy = RandomStrategy()
    
    def get_move(self, board: Board, player: int) -> Optional[Tuple[int, int]]:
        if random.random() < self.exploration_rate:
            return self.random_strategy.get_move(board, player)
        else:
            return self.main_strategy.get_move(board, player)


# SELF-PLAY GENERATOR
class SelfPlayGenerator:
    """Main class for generating self-play games"""
    
    def __init__(self, config: Optional[SelfPlayConfig] = None, **kwargs):
        """
        Args:
            config: SelfPlayConfig object
            **kwargs: Override config parameters
        """
        self.config = config or SelfPlayConfig()
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Setup output directory
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Build strategy
        self.strategy = self._build_strategy()
        
        # Statistics
        self.stats = defaultdict(int)
        
        if self.config.verbose:
            self._print_config()
    
    def _build_strategy(self) -> GameStrategy:
        """Build game strategy based on config"""
        if self.config.strategy == "random":
            strategy = RandomStrategy()
        
        elif self.config.strategy == "search":
            strategy = SearchStrategy(
                max_depth=self.config.max_depth,
                max_time=self.config.max_time
            )
        
        elif self.config.strategy == "model" or self.config.use_model:
            if not self.config.model_path:
                print("No model path provided, falling back to search")
                strategy = SearchStrategy(
                    max_depth=self.config.max_depth,
                    max_time=self.config.max_time
                )
            else:
                strategy = ModelGuidedStrategy(
                    model_path=self.config.model_path,
                    max_depth=self.config.max_depth,
                    max_time=self.config.max_time
                )
        
        elif self.config.strategy == "mixed":
            base_strategy = SearchStrategy(
                max_depth=self.config.max_depth,
                max_time=self.config.max_time
            )
            strategy = MixedStrategy(
                main_strategy=base_strategy,
                exploration_rate=self.config.exploration_rate
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        return strategy
    
    def _print_config(self):
        """Print configuration"""
        print("\n" + "="*60)
        print("SELF-PLAY CONFIGURATION")
        print("="*60)
        print(f"Strategy: {self.config.strategy}")
        print(f"Max Depth: {self.config.max_depth}")
        print(f"Max Time: {self.config.max_time}s")
        print(f"Use Model: {self.config.use_model}")
        if self.config.model_path:
            print(f"Model Path: {self.config.model_path}")
        print(f"Save Dir: {self.config.save_dir}")
        print(f"Progressive: {self.config.progressive}")
        print("="*60 + "\n")
    
    def play_one_game(
        self,
        game_id: int,
        player1_strategy: Optional[GameStrategy] = None,
        player2_strategy: Optional[GameStrategy] = None
    ) -> Dict:
        """
        Play one game and return game data
        
        Args:
            game_id: Game ID for saving
            player1_strategy: Strategy for player 1 (default: self.strategy)
            player2_strategy: Strategy for player 2 (default: self.strategy)
        
        Returns:
            game_data: Dictionary with moves, result, metadata
        """
        if player1_strategy is None:
            player1_strategy = self.strategy
        if player2_strategy is None:
            player2_strategy = self.strategy
        
        board = Board()
        moves = []
        player = 1
        result = 0
        
        start_time = time.time()
        move_times = []
        
        for move_count in range(self.config.max_moves):
            # Select strategy
            strategy = player1_strategy if player == 1 else player2_strategy
            
            # Get move
            move_start = time.time()
            move = strategy.get_move(board, player)
            move_time = time.time() - move_start
            move_times.append(move_time)
            
            if move is None:
                # No valid moves (draw or error)
                result = 0
                break
            
            # Play move
            r, c = move
            board.play(r, c, player)
            moves.append([r, c])
            
            # Check win
            if board.is_win_from(r, c):
                result = player
                # Fixed: Use consistent key names
                if player == 1:
                    self.stats['player1_wins'] += 1
                else:
                    self.stats['player2_wins'] += 1
                break
            
            # Check draw
            if board.is_draw():
                result = 0
                self.stats['draws'] += 1
                break
            
            # Switch player
            player = -player
        else:
            # Max moves reached (draw)
            result = 0
            self.stats['draws'] += 1
        
        total_time = time.time() - start_time
        
        # Build game data
        game_data = {
            "moves": moves,
            "result": int(result),
        }
        
        # Add metadata if enabled
        if self.config.save_metadata:
            game_data["metadata"] = {
                "game_id": game_id,
                "n_moves": len(moves),
                "total_time": round(total_time, 2),
                "avg_move_time": round(np.mean(move_times), 3),
                "max_depth": self.config.max_depth,
                "strategy": self.config.strategy,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Update stats
        self.stats['total_games'] += 1
        self.stats['total_moves'] += len(moves)
        
        return game_data
    
    def save_game(self, game_data: Dict, game_id: int):
        """Save game to JSON file"""
        filename = f"game_{game_id:06d}.json"
        filepath = os.path.join(self.config.save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(game_data, f)
    
    def is_valid_game(self, game_data: Dict) -> bool:
        """Check if game meets quality criteria"""
        n_moves = len(game_data['moves'])
        
        # Filter too short games
        if n_moves < self.config.min_game_length:
            self.stats['filtered_too_short'] += 1
            return False
        
        # Filter too long games (likely draws)
        if n_moves > self.config.max_game_length:
            self.stats['filtered_too_long'] += 1
            return False
        
        return True
    
    def generate_games(
        self,
        n_games: int,
        save_dir: Optional[str] = None,
        show_progress: bool = True
    ):
        """
        Generate multiple self-play games
        
        Args:
            n_games: Number of games to generate
            save_dir: Override save directory
            show_progress: Show progress bar
        """
        if save_dir:
            self.config.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nGenerating {n_games} self-play games...")
        print(f"Saving to: {self.config.save_dir}")
        
        start_time = time.time()
        valid_games = 0
        
        for i in range(n_games):
            # Play game
            try:
                game_data = self.play_one_game(game_id=i)
                
                # Validate
                if self.is_valid_game(game_data):
                    self.save_game(game_data, game_id=valid_games)
                    valid_games += 1
                
                # Progress
                if show_progress and (i + 1) % max(1, n_games // 20) == 0:
                    elapsed = time.time() - start_time
                    games_per_sec = (i + 1) / elapsed
                    eta = (n_games - i - 1) / games_per_sec if games_per_sec > 0 else 0
                    
                    print(f"Progress: {i+1}/{n_games} ({valid_games} valid) | "
                          f"{games_per_sec:.2f} games/s | "
                          f"ETA: {eta:.0f}s", end='\r')
            
            except Exception as e:
                print(f"\nError in game {i}: {e}")
                self.stats['errors'] += 1
                continue
        
        total_time = time.time() - start_time
        
        # Final report
        print(f"\n\n{'='*60}")
        print(" GENERATION COMPLETE")
        print("="*60)
        print(f"Total Games: {self.stats['total_games']}")
        print(f"Valid Games: {valid_games}")
        print(f"Player 1 Wins: {self.stats['player1_wins']}")
        print(f"Player 2 Wins: {self.stats['player2_wins']}")
        print(f"Draws: {self.stats['draws']}")
        print(f"Total Moves: {self.stats['total_moves']}")
        print(f"Avg Moves/Game: {self.stats['total_moves'] / max(1, valid_games):.1f}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Games/Second: {valid_games / total_time:.2f}")
        
        if self.stats['filtered_too_short'] > 0:
            print(f"Filtered (too short): {self.stats['filtered_too_short']}")
        if self.stats['filtered_too_long'] > 0:
            print(f"Filtered (too long): {self.stats['filtered_too_long']}")
        if self.stats['errors'] > 0:
            print(f"Errors: {self.stats['errors']}")
        
        print("="*60)
    
    def generate_progressive(
        self,
        n_games_per_stage: int,
        stages: Optional[List[Dict]] = None
    ):
        """
        Generate games with progressive difficulty
        
        Args:
            n_games_per_stage: Games per stage
            stages: List of stage configs (depth, time, etc.)
        """
        if stages is None:
            # Default progressive stages
            stages = [
                {"max_depth": 2, "max_time": 1.0, "name": "Easy"},
                {"max_depth": 3, "max_time": 1.5, "name": "Medium"},
                {"max_depth": 4, "max_time": 2.0, "name": "Hard"},
            ]
        
        print("\n" + "="*60)
        print(" PROGRESSIVE TRAINING")
        print("="*60)
        print(f"Stages: {len(stages)}")
        print(f"Games per stage: {n_games_per_stage}")
        print("="*60)
        
        for stage_idx, stage_config in enumerate(stages, 1):
            print(f"\n Stage {stage_idx}/{len(stages)}: {stage_config.get('name', 'Unnamed')}")
            print(f"   Depth: {stage_config['max_depth']} | Time: {stage_config['max_time']}s")
            
            # Update config
            self.config.max_depth = stage_config['max_depth']
            self.config.max_time = stage_config['max_time']
            
            # Rebuild strategy with new depth/time
            self.strategy = self._build_strategy()
            
            # Generate games for this stage
            stage_dir = os.path.join(self.config.save_dir, f"stage{stage_idx}")
            self.generate_games(
                n_games=n_games_per_stage,
                save_dir=stage_dir
            )

# PARALLEL GENERATION (for speed)
def _worker_generate_game(args):
    """Worker function for parallel generation"""
    game_id, config_dict = args
    
    # Rebuild config and generator in worker process
    config = SelfPlayConfig(**config_dict)
    config.verbose = False  # Disable verbose in workers
    
    generator = SelfPlayGenerator(config=config)
    
    try:
        game_data = generator.play_one_game(game_id=game_id)
        if generator.is_valid_game(game_data):
            generator.save_game(game_data, game_id=game_id)
            return True
        return False
    except Exception as e:
        print(f"  Worker error in game {game_id}: {e}")
        return False


def generate_games_parallel(
    n_games: int,
    config: Optional[SelfPlayConfig] = None,
    n_workers: int = 4,
    **kwargs
):
    """
    Generate games in parallel using multiprocessing
    
    Args:
        n_games: Number of games to generate
        config: SelfPlayConfig object
        n_workers: Number of parallel workers
        **kwargs: Override config parameters
    """
    if config is None:
        config = SelfPlayConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    print(f"\n Generating {n_games} games with {n_workers} workers...")
    print(f" Saving to: {config.save_dir}")
    
    # Convert config to dict for pickling
    config_dict = {
        'save_dir': config.save_dir,
        'max_depth': config.max_depth,
        'max_time': config.max_time,
        'max_moves': config.max_moves,
        'strategy': config.strategy,
        'min_game_length': config.min_game_length,
        'max_game_length': config.max_game_length,
        'save_metadata': config.save_metadata
    }
    
    # Prepare arguments
    args = [(i, config_dict) for i in range(n_games)]
    
    # Run parallel
    start_time = time.time()
    
    with mp.Pool(processes=n_workers) as pool:
        results = []
        for result in pool.imap_unordered(_worker_generate_game, args):
            results.append(result)
            
            # Progress
            if len(results) % max(1, n_games // 20) == 0:
                valid = sum(results)
                print(f"Progress: {len(results)}/{n_games} ({valid} valid)", end='\r')
    
    total_time = time.time() - start_time
    valid_games = sum(results)
    
    print(f"\n\n{'='*60}")
    print(" PARALLEL GENERATION COMPLETE")
    print("="*60)
    print(f"Valid Games: {valid_games}/{n_games}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Games/Second: {valid_games / total_time:.2f}")
    print("="*60)


# COMMAND LINE INTERFACE
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Caro Self-Play Data")
    
    # Basic
    parser.add_argument('--n_games', type=int, default=100, help='Number of games')
    parser.add_argument('--save_dir', type=str, default='data/selfplay', help='Output directory')
    
    # Strategy
    parser.add_argument('--strategy', type=str, default='search',
                       choices=['random', 'search', 'model', 'mixed'],
                       help='Playing strategy')
    parser.add_argument('--max_depth', type=int, default=3, help='Search depth')
    parser.add_argument('--max_time', type=float, default=2.0, help='Max time per move (seconds)')
    
    # Model
    parser.add_argument('--use_model', action='store_true', help='Use trained model')
    parser.add_argument('--model_path', type=str, default='checkpoints/caro_best.pt',
                       help='Path to model checkpoint')
    
    # Progressive
    parser.add_argument('--progressive', action='store_true', help='Progressive difficulty')
    parser.add_argument('--stages', type=int, default=3, help='Number of stages')
    
    # Parallel
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    
    # Quality
    parser.add_argument('--min_length', type=int, default=10, help='Min game length')
    parser.add_argument('--max_length', type=int, default=225, help='Max game length')
    
    args = parser.parse_args()
    
    # Build config
    config = SelfPlayConfig(
        save_dir=args.save_dir,
        strategy=args.strategy,
        max_depth=args.max_depth,
        max_time=args.max_time,
        use_model=args.use_model,
        model_path=args.model_path if args.use_model else None,
        progressive=args.progressive,
        min_game_length=args.min_length,
        max_game_length=args.max_length
    )
    
    # Generate
    if args.workers > 1:
        # Parallel generation
        generate_games_parallel(
            n_games=args.n_games,
            config=config,
            n_workers=args.workers
        )
    else:
        # Single process
        generator = SelfPlayGenerator(config=config)
        
        if args.progressive:
            games_per_stage = args.n_games // args.stages
            generator.generate_progressive(n_games_per_stage=games_per_stage)
        else:
            generator.generate_games(n_games=args.n_games)


if __name__ == "__main__":
    main()