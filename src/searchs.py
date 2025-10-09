# thuật toán Negamax + Alpha-Beta + TT


# search.py
# Negamax with alpha-beta pruning and a lightweight Transposition Table (TT)
# Designed to be readable and easy to follow (for đồ án).
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import time
import math
from typing import Optional, Tuple, Dict
from game import Board
from evaluate import evaluate_simple, evaluate_pattern

INF = 1e9

class TTEntry:
    def __init__(self, depth: int, value: float, flag: str, best_move: Optional[Tuple[int,int]]):
        self.depth = depth      # depth của entry
        self.value = value      # giá trị đã tính
        self.flag = flag        # 'EXACT' / 'LOWER' / 'UPPER'
        self.best_move = best_move

class TranspositionTable:
    def __init__(self):
        self.table: Dict[int, TTEntry] = {}

    def get(self, key: int) -> Optional[TTEntry]:
        return self.table.get(int(key), None)

    def put(self, key: int, entry: TTEntry):
        self.table[int(key)] = entry

def negamax(board: Board, depth: int, alpha: float, beta: float,
            player: int, tt: TranspositionTable,
            evaluate_fn = evaluate_pattern) -> float:
    """
    Hàm negamax trả score từ POV của 'player' (người turn hiện tại khi gọi).
    Tham số:
      - board: Board object
      - depth: số bậc còn lại
      - alpha, beta: bound
      - player: 1 hoặc -1 (người đang đi)
      - tt: TranspositionTable instance
      - evaluate_fn: function(board, player) -> float
    """
    # Check terminal by last move
    if board.last_move is not None:
        lr, lc = board.last_move
        if board.is_win_from(lr, lc):
            # nếu last move của opponent tạo thắng thì current player thua
            return -INF + 1  # a large negative value

    if depth == 0:
        return evaluate_fn(board, player)

    key = int(board.zobrist_hash)
    tt_entry = tt.get(key)
    if tt_entry and tt_entry.depth >= depth:
        # sử dụng TT: xử lý theo flag
        if tt_entry.flag == "EXACT":
            return tt_entry.value
        elif tt_entry.flag == "LOWER":
            alpha = max(alpha, tt_entry.value)
        elif tt_entry.flag == "UPPER":
            beta = min(beta, tt_entry.value)
        if alpha >= beta:
            return tt_entry.value

    best_value = -INF
    best_move = None

    # move ordering: lấy candidates và sắp theo evaluate tạm
    moves = board.generate_candidates(radius=2)
    # đánh giá sơ bộ mỗi nước để sắp xếp: nhanh, không thay đổi board lâu
    scored_moves = []
    for mv in moves:
        r,c = mv
        # tạm đánh để tính heuristic nhỏ (không update zobrist để nhanh)
        board.grid[r,c] = player
        s = evaluate_pattern(board, player)
        board.grid[r,c] = 0
        scored_moves.append((s, mv))
    # sort descending (kiếm move có score lớn trước)
    scored_moves.sort(reverse=True, key=lambda x: x[0])
    ordered_moves = [mv for (_, mv) in scored_moves]

    for mv in ordered_moves:
        r,c = mv
        board.play(r,c,player)
        val = -negamax(board, depth-1, -beta, -alpha, -player, tt, evaluate_fn)
        board.undo()
        if val > best_value:
            best_value = val
            best_move = mv
        alpha = max(alpha, val)
        if alpha >= beta:
            break  # prune

    # save to TT
    if best_value <= alpha:
        flag = "UPPER"
    elif best_value >= beta:
        flag = "LOWER"
    else:
        flag = "EXACT"
    tt.put(key, TTEntry(depth=depth, value=best_value, flag=flag, best_move=best_move))
    return best_value

def get_best_move(board: Board, max_depth: int = 3, max_time: Optional[float] = None,
                  player: int = 1, evaluate_fn = evaluate_pattern) -> Tuple[Optional[Tuple[int,int]], float]:
    """
    Iterative deepening wrapper:
      - chạy depth 1..max_depth
      - nếu max_time được set (seconds), loop sẽ dừng khi time hết
    Trả: (best_move, best_value)
    """
    start_time = time.time()
    tt = TranspositionTable()
    best_move = None
    best_value = -INF

    for d in range(1, max_depth+1):
        # time check
        if max_time is not None and (time.time() - start_time) > max_time:
            break
        # đơn giản: loop qua moves và dùng negamax
        moves = board.generate_candidates(radius=2)
        local_best_move = None
        local_best_value = -INF
        # try ordering by previous heuristic
        for mv in moves:
            r,c = mv
            board.play(r,c,player)
            if board.is_win_from(r,c):
                val = INF  # immediate win
            else:
                val = -negamax(board, d-1, -INF, INF, -player, tt, evaluate_fn)
            board.undo()
            if val > local_best_value:
                local_best_value = val
                local_best_move = mv
        # update global best if better
        if local_best_move is not None:
            best_move = local_best_move
            best_value = local_best_value
    return best_move, best_value
