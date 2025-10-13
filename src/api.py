# hàm xử lý giao diện


import numpy as np
import time
from typing import Optional, List, Tuple

from game import Board
from searchs import get_best_move

try:
    from model import load_checkpoint, load_model_into_cache, evaluate_model, policy_suggest
except Exception:
    # older/newer variants may exist; we'll attempt to import minimal APIs at runtime
    load_model_into_cache = None
    evaluate_model = None
    policy_suggest = None

class CaroController:
    """
    Controller that links Board (game logic) with AI (search+model).
    - Human = 1 (X), AI = -1 (O)
    """
    def __init__(self, model_path: Optional[str] = None, max_depth: int = 4, max_time: float = 2.0):
        self.board = Board()
        self.turn = 1
        self.winner = 0
        self.last_move = None
        self.win_line = None

        self.max_depth = max_depth
        self.max_time = max_time

        # Model loaded flags
        self.model_loaded = False
        self.model_path = model_path

        if model_path and load_model_into_cache is not None:
            try:
                # many variants in project: some files use load_model_into_cache(model_path, use_fp16=True, use_ema=True)
                load_model_into_cache(model_path, use_fp16=True, use_ema=True)
                self.model_loaded = True
                print(f"✅ Model loaded to cache: {model_path}")
            except Exception as e:
                print(f"⚠️ Model load failed: {e}. Falling back to pattern evaluate/search.")
                self.model_loaded = False

    # ---------- Game wrappers ----------
    def reset(self):
        self.board = Board()
        self.turn = 1
        self.winner = 0
        self.last_move = None
        self.win_line = None

    def is_valid_move(self, r:int, c:int) -> bool:
        return self.board.in_bounds(r, c) and self.board.grid[r,c] == 0

    def human_move(self, r:int, c:int) -> bool:
        """Place human move (X=1). Return True if placed."""
        if self.winner != 0:
            return False
        if not self.is_valid_move(r, c):
            return False
        self.board.play(r, c, 1)
        self.last_move = (r, c)
        if self.board.is_win_from(r, c):
            self.winner = 1
            self.win_line = self._get_win_line(r, c, player=1)
        elif self.board.is_draw():
            self.winner = 0
        else:
            self.turn = -1
        return True

    def ai_move(self) -> Optional[Tuple[int,int]]:
        """Ask search+model for best move for player -1 (AI)."""
        if self.winner != 0:
            return None

        # If model available, get policy_suggest to pass policy priors (optional)
        policy_scores = None
        if self.model_loaded and policy_suggest is not None:
            try:
                policy_scores = policy_suggest(self.board, top_k=40)
            except Exception:
                policy_scores = None

        print("🤖 AI thinking... (this may take a few seconds)")
        best_move, value, stats = get_best_move(
            board=self.board,
            max_depth=self.max_depth,
            max_time=self.max_time,
            player=-1,
            evaluate_fn=(evaluate_model if (self.model_loaded and evaluate_model is not None) else None),
            policy_scores=policy_scores,
            verbose=False
        )

        if best_move is None:
            # fallback: pick random legal
            moves = self.board.generate_candidates(radius=2)
            if not moves:
                moves = self.board.legal_moves()
            if not moves:
                return None
            import random
            best_move = random.choice(moves)

        r, c = best_move
        self.board.play(r, c, -1)
        self.last_move = (r, c)
        if self.board.is_win_from(r, c):
            self.winner = -1
            self.win_line = self._get_win_line(r, c, player=-1)
        elif self.board.is_draw():
            self.winner = 0
        else:
            self.turn = 1

        print(f"✅ AI played {best_move}  (value={value:.2f})")
        return best_move

    # ---------- helper to extract contiguous 5 for highlight ----------
    def _get_win_line(self, r:int, c:int, player:int) -> Optional[List[Tuple[int,int]]]:
        """
        Scan four directions from (r,c) to find the 5-in-a-row that formed the win.
        Returns list of coordinates (length >=5) or None.
        """
        grid = self.board.grid
        N = grid.shape[0]
        dirs = [(1,0),(0,1),(1,1),(1,-1)]
        for dr, dc in dirs:
            line = [(r, c)]
            # forward
            rr, cc = r+dr, c+dc
            while 0 <= rr < N and 0 <= cc < N and grid[rr,cc] == player:
                line.append((rr,cc))
                rr += dr; cc += dc
            # backward
            rr, cc = r-dr, c-dc
            while 0 <= rr < N and 0 <= cc < N and grid[rr,cc] == player:
                line.insert(0, (rr,cc))
                rr -= dr; cc -= dc
            if len(line) >= 5:
                # return the contiguous five (could be longer than 5)
                # choose the first 5 contiguous (or center 5)
                if len(line) == 5:
                    return line
                else:
                    # return middle five around (r,c)
                    idx = line.index((r,c))
                    start = max(0, idx-2)
                    return line[start:start+5]
        return None
