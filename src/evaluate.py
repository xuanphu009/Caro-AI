# evaluate.py
# Hàm đánh giá nhanh, dùng khi chưa gắn CNN.
# Trả giá trị số: càng lớn có lợi cho current_player

"""
CARO AI - BOARD EVALUATION MODULE (v3.0 FIXED)
Pattern-based heuristic evaluation for 15x15 Gomoku / Caro
Compatible with searchs.py, selfplay.py, and model.py
"""
from __future__ import annotations

import numpy as np
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from typing import Tuple, Optional, Dict, Any, List

BOARD_SIZE = 15
WIN_LEN = 5

# Trọng số cho từng loại pattern
_WEIGHTS = {
    "FIVE":      1.0e6,    # 5 liên tiếp → chiến thắng
    "OPEN4":     1.0e5,    # 4 mở hai đầu
    "HALF4":     2.0e4,    # 4 bị chặn một đầu
    "OPEN3":     3.0e3,    # 3 mở hai đầu
    "HALF3":     5.0e2,    # 3 bị chặn một đầu
    "OPEN2":     1.0e2,    # 2 mở
    "HALF2":     2.0e1,    # 2 bị chặn
    "CENTER":    5.0,      # Ưu tiên ô giữa bàn
    "MOBILITY":  1.0       # Độ linh hoạt (số ô có thể đi)
}

# Mẫu pattern để nhận dạng theo regex
_PATTERNS = {
    "FIVE": re.compile(r"X{5}"),          # 5 quân liên tiếp
    "OPEN4": re.compile(r"\.X{4}\."),     # 4 quân mở hai đầu
    "HALF4": re.compile(r"X{4}\.|\.X{4}|X{3}\.X|X\.X{3}"),
    "OPEN3": re.compile(r"\.X{3}\."),     # 3 quân mở hai đầu
    "HALF3": re.compile(r"X{3}\.|\.X{3}|X{2}\.X|X\.X{2}"),
    "OPEN2": re.compile(r"\.X{2}\."),     # 2 quân mở hai đầu
    "HALF2": re.compile(r"\.X{2}|X{2}\.|X\.X")
}

# Global model cache (for evaluate_model)
_CACHED_MODEL = None

def _to_grid(board: Any) -> np.ndarray:
    """Chuyển đổi đối tượng board thành mảng numpy 2D
        Nếu đã là numpy → ép định dạng"""
    if hasattr(board, "grid"):
        g = np.array(board.grid)
    else:
        g = np.array(board)
    if g.ndim == 0:
        g = np.reshape(g, (BOARD_SIZE, BOARD_SIZE))
    if g.shape != (BOARD_SIZE, BOARD_SIZE):
        try:
            g = g.reshape((BOARD_SIZE, BOARD_SIZE))
        except Exception:
            g = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    return g.astype(np.int8)

def _line_to_string(line: np.ndarray, player_val: int) -> str:
    """Chuyển 1 hàng/cột/đường chéo thành chuỗi:
        X: quân mình, O: quân đối, .: trống"""
    if player_val == 1:
        p_mask = (line == 1)
        o_mask = (line == -1)
    else:
        p_mask = (line == -player_val)
        o_mask = (line == player_val)
    s_chars = np.full(line.shape, '.', dtype='<U1')
    s_chars[p_mask] = 'X'
    s_chars[o_mask] = 'O'
    return ''.join(s_chars.tolist())

def _scan_line_for_patterns(line_arr: np.ndarray, player: int) -> Dict[str, int]:
    """Count pattern occurrences in a line"""
    s = _line_to_string(line_arr, player)
    counts = {}
    for name, pat in _PATTERNS.items():
        found = pat.findall(s)
        counts[name] = len(found)
    return counts

def _count_patterns_grid(grid: np.ndarray, player: int) -> Dict[str, int]:
    """Count patterns across all lines"""
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

def evaluate_pattern(board: Any, player: int = 1) -> float:
    """Pattern-based evaluation (pure heuristic, no model)"""
    return evaluate_advanced(board, player, model=None, fuse_model_weight=0.0, normalize=True)

def evaluate_simple(board: Any, player: int = 1) -> float:
    """Lightweight evaluation"""
    g = _to_grid(board)
    center = BOARD_SIZE // 2
    score = float(np.sum(g == player) - np.sum(g == -player))
    if g[center, center] == player:
        score += 2.0
    elif g[center, center] == -player:
        score -= 2.0
    return score

def _has_five(g, p):
    """Quick check for 5-in-a-row"""
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

def _candidate_count(grid: np.ndarray, radius: int = 2) -> Dict[int, int]:
    """Count candidate moves for each player"""
    N = grid.shape[0]
    mask = np.zeros_like(grid, dtype=np.bool_)
    occupied = (grid != 0)
    coords = np.argwhere(occupied)
    for (r, c) in coords:
        r0 = max(0, r - radius)
        r1 = min(N, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(N, c + radius + 1)
        mask[r0:r1, c0:c1] = True
    
    candidates = mask & (~occupied)
    p_count = 0
    o_count = 0
    coords_cand = np.argwhere(candidates)
    for (r, c) in coords_cand:
        found = False
        for d in range(1, 3):
            rr0 = max(0, r - d)
            rr1 = min(N, r + d + 1)
            cc0 = max(0, c - d)
            cc1 = min(N, c + d + 1)
            sub = grid[rr0:rr1, cc0:cc1]
            if sub.size == 0:
                continue
            vals, counts = np.unique(sub[sub != 0], return_counts=True)
            if vals.size > 0:
                owner = vals[np.argmax(counts)]
                if owner == 1:
                    p_count += 1
                else:
                    o_count += 1
                found = True
                break
    return {1: p_count, -1: o_count}

def evaluate_advanced(board: Any, player: int = 1, model=None,
                      fuse_model_weight: float = 0.35, normalize: bool = True) -> float:
    """
    Đánh giá nâng cao:
    - Dựa vào pattern
    - Cộng thêm điểm từ model nếu có
    - Có chuẩn hóa đầu ra nếu muốn
    """
    grid = _to_grid(board)

    # Terminal check
    if _has_five(grid, player):
        return float(_WEIGHTS["FIVE"])
    if _has_five(grid, -player):
        return float(-_WEIGHTS["FIVE"])

    # Count patterns
    p_counts = _count_patterns_grid(grid, player)
    o_counts = _count_patterns_grid(grid, -player)

    # Build heuristic score
    h_score = 0.0
    h_score += _WEIGHTS["OPEN4"] * (p_counts.get("OPEN4", 0) - o_counts.get("OPEN4", 0))
    h_score += _WEIGHTS["HALF4"] * (p_counts.get("HALF4", 0) - o_counts.get("HALF4", 0))
    h_score += _WEIGHTS["OPEN3"] * (p_counts.get("OPEN3", 0) - o_counts.get("OPEN3", 0))
    h_score += _WEIGHTS["HALF3"] * (p_counts.get("HALF3", 0) - o_counts.get("HALF3", 0))
    h_score += _WEIGHTS["OPEN2"] * (p_counts.get("OPEN2", 0) - o_counts.get("OPEN2", 0))
    h_score += _WEIGHTS["HALF2"] * (p_counts.get("HALF2", 0) - o_counts.get("HALF2", 0))

    # Center bonus
    center = BOARD_SIZE // 2
    if grid[center, center] == player:
        h_score += _WEIGHTS["CENTER"]
    elif grid[center, center] == -player:
        h_score -= _WEIGHTS["CENTER"]
    
    # Mobility
    candidates = _candidate_count(grid)
    h_score += _WEIGHTS["MOBILITY"] * (candidates[player] - candidates[-player])

    model_score = 0.0
    if model is not None and fuse_model_weight > 0.0:
        try:
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

    # Normalize
    if normalize:
        scaled = final / (_WEIGHTS["OPEN4"] * 2.0 + 1.0)
        final = float(np.tanh(scaled) * _WEIGHTS["OPEN4"])

    return float(final)


def set_model_cache(model):
    # Lưu model vào biến toàn cục để dùng đánh giá nhanh sau này
    global _CACHED_MODEL
    _CACHED_MODEL = model


def evaluate_model(board, current_player: int = 1) -> float:
    """
    Đánh giá bàn cờ bằng model đã lưu cache (nếu có)
    Nếu không dùng được → fallback về heuristic
    """
    if _CACHED_MODEL is None:
        raise RuntimeError("No model cached. Call set_model_cache(model) first.")
    
    grid = _to_grid(board)
    
    try:
        # Assume model has evaluate(grid, player) method
        return float(_CACHED_MODEL.evaluate(grid, current_player))
    except Exception:
        # Fallback to heuristic
        return evaluate_pattern(board, current_player)