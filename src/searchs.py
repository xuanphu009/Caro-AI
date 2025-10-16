# thuật toán Negamax + Alpha-Beta + TT

"""
- Negamax kết hợp cắt tỉa Alpha-Beta
- Principal Variation Search (PVS): tìm đường đi chính xác nhất
- Bảng ghi nhớ trạng thái (Transposition Table) với Zobrist hashing
- Chiến lược "Nước đi sát thủ" (Killer Move Heuristic)
- Chiến lược ghi nhớ lịch sử nước đi (History Heuristic)
- Phát hiện mối đe dọa & tìm kiếm yên tĩnh (Threat Detection & Quiescence Search)
- TÍCH HỢP MÔ HÌNH: Sử dụng CNN đã huấn luyện để đánh giá tốt hơn
- Tìm kiếm sâu dần (Iterative Deepening) kèm quản lý thời gian hợp lý
"""

"""
CARO AI SEARCH - COMPLETE REWRITE
Simple, clean, and strong AI search engine
- Minimax with Alpha-Beta Pruning
- Threat detection with priority ordering
- Transposition table for memoization
- Iterative deepening with time management
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import time
import numpy as np
from typing import Optional, Tuple, Dict, List
from game import Board
from evaluate import evaluate_pattern, evaluate_advanced

# Constants
INF = float('inf')
WIN_SCORE = 100000
BLOCK_SCORE = 50000

# ============================================================
# THREAT DETECTION - Priority 1: Find forcing moves
# ============================================================

def get_threats(board: Board, player: int) -> List[Tuple[int, int]]:
    """
    Find threat moves in priority order:
    1. Winning moves (5-in-a-row)
    2. Blocking opponent win
    3. Creating open 4 (double threat)
    4. Creating open 3
    """
    
    winning = []
    blocking = []
    open4 = []
    open3 = []
    
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for r in range(board.size):
        for c in range(board.size):
            if board.grid[r, c] != 0:
                continue
            
            # Test player move
            board.grid[r, c] = player
            for dr, dc in dirs:
                count = 1
                # Count forward
                rr, cc = r + dr, c + dc
                while board.in_bounds(rr, cc) and board.grid[rr, cc] == player:
                    count += 1
                    rr += dr
                    cc += dc
                # Count backward
                rr, cc = r - dr, c - dc
                while board.in_bounds(rr, cc) and board.grid[rr, cc] == player:
                    count += 1
                    rr -= dr
                    cc -= dc
                
                if count >= 5:
                    winning.append((r, c))
                elif count == 4:
                    open4.append((r, c))
                elif count == 3:
                    open3.append((r, c))
            
            board.grid[r, c] = 0
            
            # Test opponent move (blocking)
            board.grid[r, c] = -player
            for dr, dc in dirs:
                count = 1
                rr, cc = r + dr, c + dc
                while board.in_bounds(rr, cc) and board.grid[rr, cc] == -player:
                    count += 1
                    rr += dr
                    cc += dc
                rr, cc = r - dr, c - dc
                while board.in_bounds(rr, cc) and board.grid[rr, cc] == -player:
                    count += 1
                    rr -= dr
                    cc -= dc
                
                if count >= 5:
                    blocking.append((r, c))
                elif count == 4:
                    open4.append((r, c))
            
            board.grid[r, c] = 0
    
    # Return by priority
    if winning:
        return winning
    if blocking:
        return blocking
    if open4:
        return list(set(open4))
    if len(open3) <= 10:
        return open3
    return []


# ============================================================
# TRANSPOSITION TABLE - Cache evaluation results
# ============================================================

class TranspositionTable:
    def __init__(self, size: int = 10**6):
        self.table: Dict[int, Tuple[int, float]] = {}
        self.size = size
    
    def get(self, key: int) -> Optional[float]:
        if key in self.table:
            depth, value = self.table[key]
            return value
        return None
    
    def put(self, key: int, depth: int, value: float):
        if key not in self.table or self.table[key][0] < depth:
            self.table[key] = (depth, value)
    
    def clear(self):
        self.table.clear()


# ============================================================
# MAIN SEARCH ALGORITHM - Minimax with Alpha-Beta
# ============================================================

def minimax(board: Board, depth: int, alpha: float, beta: float,
            is_maximizing: bool, player: int, tt: TranspositionTable,
            evaluate_fn, max_time: float, start_time: float) -> float:
    """
    Minimax with alpha-beta pruning
    is_maximizing: True = AI (player=-1), False = Human (player=1)
    """
    
    # Time check
    if time.perf_counter() - start_time > max_time:
        return evaluate_fn(board, player)
    
    # Terminal check - did someone just win?
    if board.last_move:
        r, c = board.last_move
        if board.is_win_from(r, c):
            if is_maximizing:
                return WIN_SCORE - depth  # AI winning (higher depth = faster win = better)
            else:
                return -WIN_SCORE + depth  # Human winning (bad for AI)
    
    # Depth limit - evaluate
    if depth == 0:
        return evaluate_fn(board, player)
    
    # Transposition table lookup
    key = int(board.zobrist_hash)
    tt_value = tt.get(key)
    if tt_value is not None:
        return tt_value
    
    # Get candidate moves
    threats = get_threats(board, player if is_maximizing else -player)
    if threats:
        moves = threats
    else:
        moves = board.generate_candidates(radius=2)
    
    if not moves:
        return evaluate_fn(board, player)
    
    # Minimax search
    if is_maximizing:
        # AI's turn (maximize)
        max_value = -INF
        for r, c in moves:
            board.play(r, c, -1)
            value = minimax(board, depth - 1, alpha, beta, False, player, tt,
                          evaluate_fn, max_time, start_time)
            board.undo()
            
            max_value = max(max_value, value)
            alpha = max(alpha, value)
            
            if beta <= alpha:
                break  # Prune
        
        tt.put(key, depth, max_value)
        return max_value
    else:
        # Human's turn (minimize)
        min_value = INF
        for r, c in moves:
            board.play(r, c, 1)
            value = minimax(board, depth - 1, alpha, beta, True, player, tt,
                          evaluate_fn, max_time, start_time)
            board.undo()
            
            min_value = min(min_value, value)
            beta = min(beta, value)
            
            if beta <= alpha:
                break  # Prune
        
        tt.put(key, depth, min_value)
        return min_value


# ============================================================
# MAIN INTERFACE - Get best move with iterative deepening
# ============================================================

def get_best_move(board: Board, max_depth: int = 4, max_time: float = 2.0,
                  player: int = 1, evaluate_fn=None,
                  policy_scores: Optional[Dict] = None,
                  use_model: bool = False,
                  model_path: Optional[str] = None,
                  verbose: bool = False) -> Tuple[Optional[Tuple[int, int]], float, dict]:
    """
    Find best move for AI (player=-1)
    
    Args:
        board: Game board
        max_depth: Maximum search depth
        max_time: Maximum time in seconds
        player: Current player perspective (for evaluation)
        evaluate_fn: Custom evaluation function
        use_model: Use trained model
        model_path: Path to model
        verbose: Print debug info
    
    Returns:
        (best_move, best_value, stats)
    """
    
    # Setup evaluation function
    if evaluate_fn is None:
        if use_model and model_path:
            try:
                from model import load_model_into_cache
                engine = load_model_into_cache(model_path, use_fp16=False, use_ema=True)
                
                def eval_with_model(b, p):
                    return evaluate_advanced(b, p, model=engine, fuse_model_weight=0.5, normalize=True)
                
                evaluate_fn = eval_with_model
                if verbose:
                    print(f"[SEARCH] Loaded model: {model_path}")
            except Exception as e:
                if verbose:
                    print(f"[SEARCH] Model load failed: {e}, using heuristic")
                evaluate_fn = evaluate_pattern
        else:
            evaluate_fn = evaluate_pattern
    
    start_time = time.perf_counter()
    tt = TranspositionTable()
    
    stats = {
        'nodes': 0,
        'depth_reached': 0,
        'time_used': 0.0
    }
    
    best_move = None
    best_value = -INF
    
    # Iterative deepening
    for depth in range(1, max_depth + 1):
        if verbose:
            print(f"[SEARCH] Depth {depth}...", end='')
        
        # Get threatening moves first (highest priority)
        threats = get_threats(board, -1)
        if threats:
            candidates = threats
        else:
            candidates = board.generate_candidates(radius=2)
        
        if not candidates:
            if verbose:
                print(" No candidates")
            break
        
        depth_best_move = None
        depth_best_value = -INF
        
        # Try each candidate move
        for r, c in candidates:
            # Time check
            if time.perf_counter() - start_time > max_time:
                break
            
            # Play AI move
            board.play(r, c, -1)
            
            # Check immediate win
            if board.is_win_from(r, c):
                best_move = (r, c)
                best_value = WIN_SCORE
                board.undo()
                stats['depth_reached'] = depth
                stats['time_used'] = time.perf_counter() - start_time
                if verbose:
                    print(f" WINNING MOVE: {best_move}")
                return best_move, best_value, stats
            
            # Search
            value = minimax(board, depth - 1, -INF, INF, False, -1, tt,
                          evaluate_fn, max_time, start_time)
            board.undo()
            
            if value > depth_best_value:
                depth_best_value = value
                depth_best_move = (r, c)
        
        # Update best
        if depth_best_move is not None:
            best_move = depth_best_move
            best_value = depth_best_value
            stats['depth_reached'] = depth
        
        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f" Best: {best_move}, Value: {best_value:.1f}, Time: {elapsed:.2f}s")
        
        if elapsed > max_time:
            break
    
    stats['time_used'] = time.perf_counter() - start_time
    
    if verbose:
        print(f"[SEARCH] COMPLETE: move={best_move}, value={best_value:.1f}, time={stats['time_used']:.2f}s")
    
    return best_move, best_value, stats


# def negamax(board: Board, depth: int, alpha: float, beta: float,
#             player: int, tt: TranspositionTable, evaluate_fn) -> float:
#     # Wrapper for backward compatibility, gọi PVS với thông số mặc định
#     killers = KillerMoves()
#     history = HistoryTable(board.size)
#     stats = {'nodes': 0, 'cutoffs': 0, 'tt_hits': 0, 'tt_cutoffs': 0, 're_searches': 0}
#     return pvs(board, depth, alpha, beta, player, tt, killers, history, 
#                evaluate_fn, stats)