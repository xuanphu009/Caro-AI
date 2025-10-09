# evaluate.py
# Hàm đánh giá nhanh, dùng khi chưa gắn CNN.
# Trả giá trị số: càng lớn có lợi cho current_player.

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from game import Board
from typing import Tuple

def evaluate_simple(board: Board, current_player: int) -> float:
    """
    Rất đơn giản: số quân của current_player - số quân đối phương.
    Chỉ để test search pipeline; sẽ thay bằng pattern-based hoặc CNN.
    """
    my = int((board.grid == current_player).sum())
    opp = int((board.grid == -current_player).sum())
    return float(my - opp)

def evaluate_pattern(board: Board, current_player: int) -> float:
    """
    Heuristic pattern-based (đếm chuỗi liên tiếp).
    Mục tiêu: phân biệt 2/3/4-in-row. Không quá phức tạp nhưng hiệu quả hơn evaluate_simple.
    """
    weights = {2: 1.0, 3: 5.0, 4: 50.0}
    total = 0.0
    size = board.size
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for r in range(size):
        for c in range(size):
            player = board.grid[r,c]
            if player == 0:
                continue
            for dr,dc in dirs:
                cnt = 1
                rr,cc = r+dr, c+dc
                while board.in_bounds(rr,cc) and board.grid[rr,cc] == player:
                    cnt += 1
                    rr += dr; cc += dc
                if cnt in weights:
                    if player == current_player:
                        total += weights[cnt]
                    else:
                        total -= weights[cnt]
    return float(total)
