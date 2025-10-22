# ...existing code...
"""
Self-play generator integrated with project's search (GomokuAI) and model (CaroNet) APIs.

Design decisions:
- Use GomokuAI (source.search.GomokuAI) for alpha-beta search (no external get_best_move assumed).
- Provide a lightweight Board wrapper that the model API understands (.grid, to_cnn_input, generate_candidates, legal_moves, play, is_win_from, is_draw).
- When using model-guided play, load model via load_model_into_cache() and use policy_suggest() to bias candidate ordering.
- Save games as professional_game_XXXX.json and include "first_player" field for dataset compatibility.
"""
import os
import math
import sys
import json
import time
import random
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

# Ensure project package on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project search/model APIs
from source.search import GomokuAI, N as BOARD_N
from source.model import load_model_into_cache, evaluate_model, policy_suggest

# Fallbacks if model APIs missing
_MODEL_AVAILABLE = True
try:
    _ = load_model_into_cache, evaluate_model, policy_suggest
except Exception:
    _MODEL_AVAILABLE = False

@dataclass
class SelfPlayConfig:
    save_dir: str = "data/professional"
    n_games: int = 100
    max_depth: int = 4
    max_time: float = 2.0
    max_moves: int = 225
    strategy: str = "search"        # 'search' | 'model' | 'random' | 'mixed'
    model_path: Optional[str] = None
    exploration_rate: float = 0.08
    min_game_length: int = 8
    max_game_length: int = 225
    randomize_first_player: bool = True
    verbose: bool = True
    save_metadata: bool = True

# Minimal Board wrapper to interface with model & search
class Board:
    def __init__(self, N: int = BOARD_N):
        self.N = N
        self.grid = [[0 for _ in range(N)] for _ in range(N)]
        self.last_move = None  # (r,c,player)

    def play(self, r: int, c: int, player: int):
        if not (0 <= r < self.N and 0 <= c < self.N):
            raise IndexError("Move outside board")
        if self.grid[r][c] != 0:
            raise ValueError("Cell not empty")
        self.grid[r][c] = player
        self.last_move = (r, c, player)

    def legal_moves(self) -> List[Tuple[int,int]]:
        return [(i, j) for i in range(self.N) for j in range(self.N) if self.grid[i][j] == 0]

    def generate_candidates(self, radius: int = 2) -> List[Tuple[int,int]]:
        occupied = [(i,j) for i in range(self.N) for j in range(self.N) if self.grid[i][j] != 0]
        if not occupied:
            return [(self.N//2, self.N//2)]
        cand = set()
        for (i,j) in occupied:
            for di in range(-radius, radius+1):
                for dj in range(-radius, radius+1):
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.N and 0 <= nj < self.N and self.grid[ni][nj] == 0:
                        cand.add((ni,nj))
        return list(cand) if cand else self.legal_moves()

    def is_win_from(self, r:int, c:int) -> bool:
        # use GomokuAI.isFive logic by creating a temp GomokuAI and copying grid
        ai = GomokuAI(depth=1)
        ai.boardMap = [row.copy() for row in self.grid]
        player = ai.boardMap[r][c] if (0 <= r < self.N and 0 <= c < self.N) else 0
        if player == 0:
            return False
        return ai.isFive(r, c, player)

    def is_draw(self) -> bool:
        return all(self.grid[i][j] != 0 for i in range(self.N) for j in range(self.N))

    def to_cnn_input(self, current_player: int = 1) -> np.ndarray:
        arr = np.array(self.grid, dtype=np.int8)
        p_layer = (arr == current_player).astype(np.float32)
        o_layer = (arr == -current_player).astype(np.float32)
        return np.stack([p_layer, o_layer], axis=0)

# Strategy implementations
class GameStrategy:
    def get_move(self, board: Board, player: int) -> Optional[Tuple[int,int]]:
        raise NotImplementedError

class RandomStrategy(GameStrategy):
    def get_move(self, board: Board, player: int) -> Optional[Tuple[int,int]]:
        moves = board.generate_candidates(radius=2)
        if not moves:
            moves = board.legal_moves()
        return random.choice(moves) if moves else None

class SearchStrategy(GameStrategy):
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def build_bound_from_board(self, board: Board) -> Dict[Tuple[int,int], float]:
        # heuristic bound: use simple pattern scores by scanning neighbors (zero-initialized)
        candidates = board.generate_candidates(radius=2)
        if not candidates:
            candidates = board.legal_moves()
        # initialize with small random noise to break ties and add variation
        return {pos: float(random.uniform(-1e-3, 1e-3)) for pos in candidates}

    def get_move(self, board: Board, player: int) -> Optional[Tuple[int,int]]:
        # Use GomokuAI search by copying board into ai.boardMap
        ai = GomokuAI(depth=self.max_depth)
        # copy grid safely
        ai.boardMap = [row.copy() for row in board.grid]
        ai.depth = self.max_depth
        # initialize bound with tiny jitter to avoid deterministic tie-breaking
        bound = self.build_bound_from_board(board)
        # add tiny random perturbation to bound values each call
        for k in list(bound.keys()):
            bound[k] = bound[k] + random.uniform(-1e-3, 1e-3)
        # run alpha-beta (ai will set currentI/currentJ at top depth)
        try:
            ai.alphaBetaPruning(ai.depth, ai.boardValue, bound, -math.inf, math.inf, True if player==1 else False)
            ci = getattr(ai, "currentI", None)
            cj = getattr(ai, "currentJ", None)
            if isinstance(ci, int) and isinstance(cj, int) and 0 <= ci < len(ai.boardMap) and 0 <= cj < len(ai.boardMap[0]):
                return (ci, cj)
        except Exception:
            pass
        # fallback - choose randomly among candidates to introduce variety
        candidates = list(bound.keys()) if bound else board.generate_candidates(radius=2)
        if not candidates:
            candidates = board.legal_moves()
        return random.choice(candidates) if candidates else None

class ModelGuidedStrategy(GameStrategy):
    def __init__(self, model_path: Optional[str], max_depth: int = 4):
        self.max_depth = max_depth
        self.model_loaded = False
        self.model_path = model_path
        # try flexible load; some versions accept kwargs, some don't
        if model_path and _MODEL_AVAILABLE and load_model_into_cache is not None:
            try:
                try:
                    load_model_into_cache(model_path, use_fp16=True, use_ema=True)
                except TypeError:
                    load_model_into_cache(model_path)
                self.model_loaded = True
            except Exception as e:
                print(f"⚠️  load_model_into_cache failed: {e}")
                self.model_loaded = False

    def _get_policy_scores(self, board: Board) -> Optional[Dict[Tuple[int,int], float]]:
        if not self.model_loaded or policy_suggest is None:
            return None
        # policy_suggest might accept Board or numpy input; try both
        try:
            return policy_suggest(board, top_k=50)
        except TypeError:
            try:
                return policy_suggest(board.to_cnn_input())
            except Exception:
                return None
        except Exception:
            return None

    def get_move(self, board: Board, player: int) -> Optional[Tuple[int,int]]:
        # get policy suggestions if model available
        policy_scores = self._get_policy_scores(board)

        # Build bound from candidates and optionally weight with policy
        candidates = board.generate_candidates(radius=2)
        if not candidates:
            candidates = board.legal_moves()
        bound = {}
        for pos in candidates:
            # policy_scores may map (i,j) -> score or flattened index; handle common cases
            val = 0.0
            if isinstance(policy_scores, dict):
                val = float(policy_scores.get(pos, 0.0))
            elif isinstance(policy_scores, (list, np.ndarray)):
                # try to map pos to flat index
                try:
                    idx = pos[0] * board.N + pos[1]
                    val = float(policy_scores[idx])
                except Exception:
                    val = 0.0
            # add tiny noise to break ties and encourage variation
            bound[pos] = val + random.uniform(-1e-3, 1e-3)

        # run GomokuAI search
        ai = GomokuAI(depth=self.max_depth)
        ai.boardMap = [row.copy() for row in board.grid]
        ai.depth = self.max_depth
        try:
            ai.alphaBetaPruning(ai.depth, ai.boardValue, bound, -math.inf, math.inf, True if player==1 else False)
            ci = getattr(ai, "currentI", None)
            cj = getattr(ai, "currentJ", None)
            if isinstance(ci, int) and isinstance(cj, int) and 0 <= ci < len(ai.boardMap) and 0 <= cj < len(ai.boardMap[0]):
                return (ci, cj)
        except Exception:
            pass
        # fallback to search or random; prefer search but with exploration
        return SearchStrategy(self.max_depth).get_move(board, player)

class MixedStrategy(GameStrategy):
    def __init__(self, main_strategy: GameStrategy, exploration_rate: float = 0.08):
        self.main = main_strategy
        self.explore = RandomStrategy()
        self.p = exploration_rate

    def get_move(self, board: Board, player: int) -> Optional[Tuple[int,int]]:
        if random.random() < self.p:
            return self.explore.get_move(board, player)
        return self.main.get_move(board, player)

# Self-play generator
class SelfPlayGenerator:
    def __init__(self, config: Optional[SelfPlayConfig] = None):
        self.config = config or SelfPlayConfig()
        os.makedirs(self.config.save_dir, exist_ok=True)
        # build different strategies for player1 and player2 to avoid symmetric/play-repeat
        self.player1_strategy = self._make_strategy(is_opponent=False)
        self.player2_strategy = self._make_strategy(is_opponent=True)
        # default 'strategy' kept for backward compatibility
        self.strategy = self.player1_strategy
        self.stats = defaultdict(int)

    def _make_strategy(self, is_opponent: bool = False) -> GameStrategy:
        s = self.config.strategy
        # opponent gets slightly higher exploration to diversify games
        opp_explore = min(0.35, self.config.exploration_rate * (2.0 if is_opponent else 1.0))
        if s == 'random':
            return RandomStrategy()
        if s == 'model' and self.config.model_path and _MODEL_AVAILABLE:
            # opponent model-guided uses same model but will get jitter from bound
            return ModelGuidedStrategy(self.config.model_path, max_depth=self.config.max_depth)
        base = SearchStrategy(max_depth=self.config.max_depth)
        if s == 'mixed':
            return MixedStrategy(base, exploration_rate=opp_explore)
        # default: give opponent a mixed variant to reduce mirror games
        if is_opponent:
            return MixedStrategy(base, exploration_rate=opp_explore)
        return base

    def play_one(self, game_id: int) -> Dict:
        board = Board()
        moves: List[Tuple[int,int]] = []
        first_player = 1 if not self.config.randomize_first_player else random.choice([1, -1])
        player = first_player
        winner = 0
        move_times: List[float] = []

        for step in range(self.config.max_moves):
            # use separate strategies per side
            strat = self.player1_strategy if player == 1 else self.player2_strategy
            start = time.time()
            mv = strat.get_move(board, player)
            move_times.append(time.time() - start)
            if mv is None:
                winner = 0
                break
            r, c = int(mv[0]), int(mv[1])
            try:
                board.play(r, c, player)
            except Exception:
                # illegal move -> opponent wins
                winner = -player
                break
            moves.append([r, c])

            # check win/draw
            try:
                if board.is_win_from(r, c):
                    winner = player
                    break
                if board.is_draw():
                    winner = 0
                    break
            except Exception:
                pass

            player = -player
        else:
            winner = 0

        data = {
            "moves": moves,
            "winner": int(winner),
            "first_player": int(first_player)
        }
        if self.config.save_metadata:
            data["metadata"] = {
                "game_id": int(game_id),
                "n_moves": len(moves),
                "avg_move_time": float(np.mean(move_times)) if move_times else 0.0,
                "strategy": self.config.strategy,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        return data

    def save(self, game_data: Dict, idx: int):
        fname = f"professional_game_{idx:04d}.json"
        path = os.path.join(self.config.save_dir, fname)
        with open(path, 'w') as f:
            json.dump(game_data, f, indent=2)

    def generate(self, n_games: Optional[int] = None, start_idx: int = 0):
        n = n_games if n_games is not None else self.config.n_games
        saved = 0
        attempted = 0
        start_time = time.time()

        saved_files = []
        moves_lengths = []
        move_times = []
        p1_wins = 0
        p2_wins = 0
        draws = 0

        for i in range(n):
            attempted += 1
            gd = self.play_one(game_id=start_idx + i)
            nm = len(gd.get('moves', []))
            md = gd.get('metadata', {})
            avg_mt = md.get('avg_move_time', None)
            if avg_mt is not None:
                move_times.append(float(avg_mt))

            if nm < self.config.min_game_length or nm > self.config.max_game_length:
                self.stats['filtered'] = self.stats.get('filtered', 0) + 1
                if self.config.verbose and ((i+1) % max(1, n//20) == 0):
                    print(f"Progress: {i+1}/{n} (saved {saved}) | filtered")
                continue

            self.save(gd, start_idx + saved)
            fname = f"professional_game_{(start_idx + saved):04d}.json"
            saved_files.append(fname)
            saved += 1
            moves_lengths.append(nm)

            winner = int(gd.get('winner', 0))
            first_player = int(gd.get('first_player', 1))
            if winner == 0:
                draws += 1
                self.stats['draws'] = self.stats.get('draws', 0) + 1
            else:
                if winner == first_player:
                    p1_wins += 1
                    self.stats['player1_wins'] = self.stats.get('player1_wins', 0) + 1
                else:
                    p2_wins += 1
                    self.stats['player2_wins'] = self.stats.get('player2_wins', 0) + 1

            if self.config.verbose and ((i+1) % max(1, n//20) == 0):
                elapsed = time.time() - start_time
                print(f"Progress: {i+1}/{n} (saved {saved}) | elapsed {elapsed:.1f}s", end='\r')

        total_saved = saved
        avg_moves = float(np.mean(moves_lengths)) if moves_lengths else 0.0
        median_moves = float(np.median(moves_lengths)) if moves_lengths else 0.0
        avg_move_time = float(np.mean(move_times)) if move_times else 0.0

        buckets = defaultdict(int)
        for L in moves_lengths:
            b = (L - 1) // 10
            buckets[f"{b*10+1}-{(b+1)*10}"] += 1

        summary = {
            "attempted": attempted,
            "saved": total_saved,
            "saved_files": saved_files,
            "player1_wins": p1_wins,
            "player2_wins": p2_wins,
            "draws": draws,
            "avg_moves": avg_moves,
            "median_moves": median_moves,
            "moves_histogram": dict(buckets),
            "avg_move_time": avg_move_time,
            "config": {
                "strategy": self.config.strategy,
                "max_depth": self.config.max_depth,
                "n_games_requested": n,
                "min_game_length": self.config.min_game_length,
                "max_game_length": self.config.max_game_length
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if self.config.verbose:
            print("\n" + "="*60)
            print("SELF-PLAY GENERATION SUMMARY")
            print("="*60)
            print(f"Attempted: {attempted} | Saved: {total_saved}")
            print(f"P1 wins: {p1_wins} | P2 wins: {p2_wins} | Draws: {draws}")
            print(f"Avg moves: {avg_moves:.2f} | Median moves: {median_moves:.1f} | Avg move time: {avg_move_time:.3f}s")
            print("Moves length buckets:", dict(buckets))
            print("="*60)

        summary_path = os.path.join(self.config.save_dir, f"selfplay_summary_{time.strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(summary_path, 'w') as sf:
                json.dump(summary, sf, indent=2)
            if self.config.verbose:
                print(f"Saved summary -> {summary_path}")
        except Exception as e:
            print(f"⚠️  Failed to write summary: {e}")


# CLI entry
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--out', type=str, default='data/selfplay')
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--strategy', type=str, default='search', choices=['search','model','random','mixed'])
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--no-random-first', action='store_true')
    args = p.parse_args()

    cfg = SelfPlayConfig(
        save_dir=args.out,
        n_games=args.n,
        max_depth=args.depth,
        strategy=args.strategy,
        model_path=args.model,
        randomize_first_player=not args.no_random_first
    )
    gen = SelfPlayGenerator(cfg)
    gen.generate(n_games=cfg.n_games)

if __name__ == "__main__":
    main()
# ...existing code...