# evaluate.py
# Hàm đánh giá nhanh, dùng khi chưa gắn CNN.
# Trả giá trị số: càng lớn có lợi cho current_player

"""

CARO AI - BOARD EVALUATION MODULE (v3.0)
Pattern-based heuristic evaluation for 15x15 Gomoku / Caro
Compatible with searchs.py, selfplay.py, and model.py

AlphaZero-style advanced evaluation for Caro / Gomoku (Pro)
- Fast, robust pattern detection (open4, half4, open3, etc.)
- Feature tensor generator compatible with CNN training
- Optional model fusion (if your model exposes evaluate_model(grid, player))
- Safe handling of Board objects and numpy arrays
- Normalized output suitable for alpha-beta / negamax
"""
from __future__ import annotations

import numpy as np
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from typing import Tuple, Optional, Dict, Any, List

# Try to import model evaluation if present
try:
    from src.model import evaluate_model  # must accept (grid, player) -> scalar
except Exception:
    try:
        # try relative
        from model import evaluate_model
    except Exception:
        evaluate_model = None

# Board constants (adjust if your board size differs)
BOARD_SIZE = 15
WIN_LEN = 5

# Pattern weights (tunable)
_WEIGHTS = {
    "FIVE":      1.0e6,
    "OPEN4":     1.0e5,
    "HALF4":     2.0e4,
    "OPEN3":     3.0e3,
    "HALF3":     5.0e2,
    "OPEN2":     1.0e2,
    "HALF2":     2.0e1,
    "CENTER":    5.0,     # small center control bonus
    "MOBILITY":  1.0      # number of candidates (small)
}

# Precompile regex patterns for string-scanning lines
# We'll represent: 'X' for player, 'O' for opponent, '.' for empty
_PATTERNS = {
    # five
    "FIVE": re.compile(r"X{5}"),
    # open four: .XXXX.
    "OPEN4": re.compile(r"\.X{4}\."),
    # half four: XXXX. or .XXXX or X.XXX (some variants) - we'll keep simple
    "HALF4": re.compile(r"X{4}\.|\.X{4}|X{3}\.X|X\.X{3}"),
    # open three: ..XXX.. or .X{3}.
    "OPEN3": re.compile(r"\.X{3}\."),
    # half three: X{3}\. or \.X{3} or X\.XX etc (approx)
    "HALF3": re.compile(r"X{3}\.|\.X{3}|X{2}\.X|X\.X{2}"),
    # open two
    "OPEN2": re.compile(r"\.X{2}\."),
    # half two
    "HALF2": re.compile(r"\.X{2}|X{2}\.|X\.X")
}

# Utility --------------------------------------------------------------------

def _to_grid(board: Any) -> np.ndarray:
    """
    Convert a Board-like or numpy array to a 2D numpy int8 array shape (N,N).
    Board object expected to expose `.grid` (numpy) or indexing via .grid[r,c]
    """
    if hasattr(board, "grid"):
        g = np.array(board.grid)
    else:
        g = np.array(board)
    if g.ndim == 0:
        g = np.reshape(g, (BOARD_SIZE, BOARD_SIZE))
    # ensure integer dtype and shape
    if g.shape != (BOARD_SIZE, BOARD_SIZE):
        # try to reshape if possible
        try:
            g = g.reshape((BOARD_SIZE, BOARD_SIZE))
        except Exception:
            # fallback: create empty board if input invalid
            g = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    return g.astype(np.int8)


def _line_to_string(line: np.ndarray, player_val: int) -> str:
    """
    Convert a 1D numpy int array to string representation relative to player:
    - 'X' for player's stones, 'O' for opponent, '.' for empty
    """
    # map values efficiently
    # build string list
    if player_val == 1:
        p_mask = (line == 1)
        o_mask = (line == -1)
    else:
        p_mask = (line == -player_val)  # when player is -1, map their stones to X
        o_mask = (line == player_val)
    s_chars = np.full(line.shape, '.', dtype='<U1')
    s_chars[p_mask] = 'X'
    s_chars[o_mask] = 'O'
    return ''.join(s_chars.tolist())


def _extract_all_lines(grid: np.ndarray) -> List[str]:
    """Return list of all lines (rows, cols, diagonals) as strings of original values"""
    N = grid.shape[0]
    lines = []
    # rows
    for r in range(N):
        lines.append(''.join(map(str, grid[r, :])))
    # cols
    for c in range(N):
        lines.append(''.join(map(str, grid[:, c])))
    # diag (top-left to bottom-right)
    for k in range(-N+1, N):
        diag = np.diagonal(grid, offset=k)
        if diag.size >= WIN_LEN:
            lines.append(''.join(map(str, diag)))
    # anti-diag (top-right to bottom-left)
    flipped = np.fliplr(grid)
    for k in range(-N+1, N):
        diag = np.diagonal(flipped, offset=k)
        if diag.size >= WIN_LEN:
            lines.append(''.join(map(str, diag)))
    return lines


def _scan_line_for_patterns(line_arr: np.ndarray, player: int) -> Dict[str, int]:
    """
    Given a 1D numeric line (values -1,0,1), convert it relative to player and count
    occurrences of patterns using regex. Returns dict of counts.
    """
    s = _line_to_string(line_arr, player)
    counts = {}
    for name, pat in _PATTERNS.items():
        found = pat.findall(s)
        counts[name] = len(found)
    return counts


# Feature tensor --------------------------------------------------------------

def feature_tensor(board: Any, player: int = 1, include_history: bool = False) -> np.ndarray:
    """
    Build feature tensor similar to AlphaZero input:
    channels:
      0: current player's stones (1/0)
      1: opponent stones (1/0)
      2: empty mask (1 if empty)
      optional extra channels can be added later (move history, turn)
    Returns: np.float32 array shape (C, N, N)
    """
    grid = _to_grid(board)
    C = 3
    H = W = grid.shape[0]
    out = np.zeros((C, H, W), dtype=np.float32)
    out[0] = (grid == player).astype(np.float32)
    out[1] = (grid == -player).astype(np.float32)
    out[2] = (grid == 0).astype(np.float32)
    # optional: small center channel or move history can be appended by caller
    return out


# Advanced pattern evaluator --------------------------------------------------

def _count_patterns_grid(grid: np.ndarray, player: int) -> Dict[str, int]:
    """
    Count patterns across all lines from player's perspective.
    Uses regex patterns on stringified lines for robust matching.
    """
    N = grid.shape[0]
    total = {k: 0 for k in _PATTERNS.keys()}
    # rows
    for r in range(N):
        counts = _scan_line_for_patterns(grid[r, :], player)
        for k, v in counts.items():
            total[k] += v
    # cols
    for c in range(N):
        counts = _scan_line_for_patterns(grid[:, c], player)
        for k, v in counts.items():
            total[k] += v
    # diag
    for k in range(-N+1, N):
        diag = np.diagonal(grid, offset=k)
        if diag.size >= WIN_LEN:
            counts = _scan_line_for_patterns(diag, player)
            for kk, vv in counts.items():
                total[kk] += vv
    # anti-diag
    flipped = np.fliplr(grid)
    for k in range(-N+1, N):
        diag = np.diagonal(flipped, offset=k)
        if diag.size >= WIN_LEN:
            counts = _scan_line_for_patterns(diag, player)
            for kk, vv in counts.items():
                total[kk] += vv
    return total


def evaluate_advanced(board: Any, player: int = 1, model: Optional[Any] = None,
                      fuse_model_weight: float = 0.35, normalize: bool = True) -> float:
    """
    Advanced evaluation combining pattern-based heuristics + optional model fusion.

    Args:
        board: Board object or ndarray (N,N)
        player: which player to evaluate for (1 or -1)
        model: optional model object / function - if None uses evaluate_model if available
               expected interface: evaluate_model(grid, player) -> scalar
        fuse_model_weight: how much weight to give model score (0..1)
        normalize: whether to squash final score to a bounded range

    Returns:
        float score (positive => good for `player`, negative => bad)
    """
    grid = _to_grid(board)

    # Fast terminal check (win/loss)
    # check player win
    def _has_five(g, p):
        # quick sliding check
        N = g.shape[0]
        # rows
        for r in range(N):
            row = g[r, :]
            if np.any(np.convolve((row == p).astype(int), np.ones(5, dtype=int), mode='valid') >= 5):
                return True
        # cols
        for c in range(N):
            col = g[:, c]
            if np.any(np.convolve((col == p).astype(int), np.ones(5, dtype=int), mode='valid') >= 5):
                return True
        # diags
        for k in range(-N+1, N):
            diag = np.diagonal(g, offset=k)
            if diag.size >= WIN_LEN:
                if np.any(np.convolve((diag == p).astype(int), np.ones(5, dtype=int), mode='valid') >= 5):
                    return True
        # anti-diags
        ff = np.fliplr(g)
        for k in range(-N+1, N):
            diag = np.diagonal(ff, offset=k)
            if diag.size >= WIN_LEN:
                if np.any(np.convolve((diag == p).astype(int), np.ones(5, dtype=int), mode='valid') >= 5):
                    return True
        return False

    if _has_five(grid, player):
        return float(_WEIGHTS["FIVE"])
    if _has_five(grid, -player):
        return float(-_WEIGHTS["FIVE"])

    # Count patterns for player and opponent
    p_counts = _count_patterns_grid(grid, player)
    o_counts = _count_patterns_grid(grid, -player)

    # Build heuristic score as weighted sum
    h_score = 0.0
    h_score += _WEIGHTS["OPEN4"] * (p_counts.get("OPEN4", 0) - o_counts.get("OPEN4", 0))
    h_score += _WEIGHTS["HALF4"] * (p_counts.get("HALF4", 0) - o_counts.get("HALF4", 0))
    h_score += _WEIGHTS["OPEN3"] * (p_counts.get("OPEN3", 0) - o_counts.get("OPEN3", 0))
    h_score += _WEIGHTS["HALF3"] * (p_counts.get("HALF3", 0) - o_counts.get("HALF3", 0))
    h_score += _WEIGHTS["OPEN2"] * (p_counts.get("OPEN2", 0) - o_counts.get("OPEN2", 0))
    h_score += _WEIGHTS["HALF2"] * (p_counts.get("HALF2", 0) - o_counts.get("HALF2", 0))

    # small center bonus and mobility
    center = BOARD_SIZE // 2
    if grid[center, center] == player:
        h_score += _WEIGHTS["CENTER"]
    elif grid[center, center] == -player:
        h_score -= _WEIGHTS["CENTER"]
    # mobility: number of candidate moves near stones (cheap)
    candidates = _candidate_count(grid)
    # positive if many available
    h_score += _WEIGHTS["MOBILITY"] * (candidates[player] - candidates[-player])

    # Model fusion (optional)
    model_score = 0.0
    if model is None and evaluate_model is not None:
        try:
            model_score = float(evaluate_model(grid, player))
        except Exception:
            model_score = 0.0
    elif model is not None:
        try:
            # model can be a function or object exposing evaluate_model
            if callable(model):
                model_score = float(model(grid, player))
            else:
                model_score = float(model.evaluate(grid, player))
        except Exception:
            model_score = 0.0

    if model_score != 0.0 and fuse_model_weight > 0.0:
        final = (1.0 - fuse_model_weight) * h_score + fuse_model_weight * model_score
    else:
        final = h_score

    # Normalize / squash to bounded range to avoid gigantic values in search
    if normalize:
        # scale by a heuristic factor and squash with tanh
        scaled = final / (_WEIGHTS["OPEN4"] * 2.0 + 1.0)
        final = float(np.tanh(scaled) * _WEIGHTS["OPEN4"])

    return float(final)


def _candidate_count(grid: np.ndarray, radius: int = 2) -> Dict[int, int]:
    """
    Count candidate squares near existing stones for both players.
    Returns dict {player: count, -player: count}
    """
    N = grid.shape[0]
    mask = np.zeros_like(grid, dtype=np.bool_)
    occupied = (grid != 0)
    # for each occupied cell, mark neighbors within radius
    coords = np.argwhere(occupied)
    for (r, c) in coords:
        r0 = max(0, r - radius); r1 = min(N, r + radius + 1)
        c0 = max(0, c - radius); c1 = min(N, c + radius + 1)
        mask[r0:r1, c0:c1] = True
    # candidate = mask & empty
    candidates = mask & (~occupied)
    # simple heuristic: count how many candidate positions are nearer to each player's stones
    p_count = 0
    o_count = 0
    coords_cand = np.argwhere(candidates)
    for (r, c) in coords_cand:
        # compute nearest occupied stone owner (small radius search)
        found = False
        for d in range(1, 3):
            rr0 = max(0, r - d); rr1 = min(N, r + d + 1)
            cc0 = max(0, c - d); cc1 = min(N, c + d + 1)
            sub = grid[rr0:rr1, cc0:cc1]
            if sub.size == 0:
                continue
            vals, counts = np.unique(sub[sub != 0], return_counts=True)
            if vals.size > 0:
                # choose majority in neighborhood
                owner = vals[np.argmax(counts)]
                if owner == 1:
                    p_count += 1
                else:
                    o_count += 1
                found = True
                break
        if not found:
            # neutral bump both a little
            p_count += 0
            o_count += 0
    return {1: p_count, -1: o_count}


# Backwards compat helpers
def evaluate_pattern(board: Any, player: int = 1) -> float:
    """Backward compatible wrapper mapping to advanced pattern-only eval."""
    return evaluate_advanced(board, player, model=None, fuse_model_weight=0.0, normalize=True)


def evaluate_simple(board: Any, player: int = 1) -> float:
    """Lightweight wrapper (stone difference + center bias)."""
    g = _to_grid(board)
    center = BOARD_SIZE // 2
    score = float(np.sum(g == player) - np.sum(g == -player))
    if g[center, center] == player:
        score += 2.0
    elif g[center, center] == -player:
        score -= 2.0
    return score
