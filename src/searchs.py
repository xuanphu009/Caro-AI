# thuật toán Negamax + Alpha-Beta + TT

"""
Advanced Search Engine for Caro (Gomoku) with multiple optimizations:
- Negamax with Alpha-Beta Pruning
- Principal Variation Search (PVS)
- Transposition Table with Zobrist hashing
- Killer Move Heuristic
- History Heuristic
- Aspiration Windows
- Late Move Reduction (LMR)
- Threat Detection & Quiescence Search
- Policy Network Integration (optional)
- Iterative Deepening with time management
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import time
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import defaultdict
from game import Board
from evaluate import evaluate_pattern, evaluate_simple
from model import evaluate_model

# CONSTANTS
INF = float('inf')
MATE_SCORE = 1000000
MAX_DEPTH = 50

# TRANSPOSITION TABLE
class TTEntry:
    """Entry trong Transposition Table với đầy đủ thông tin"""
    __slots__ = ['depth', 'value', 'flag', 'best_move', 'age']
    
    def __init__(self, depth: int, value: float, flag: str, 
                 best_move: Optional[Tuple[int,int]], age: int = 0):
        self.depth = depth
        self.value = value
        self.flag = flag  # 'EXACT' / 'LOWER' / 'UPPER'
        self.best_move = best_move
        self.age = age

class TranspositionTable:
    """TT with aging and size limit for memory efficiency"""
    def __init__(self, max_size: int = 10**6):
        self.table: Dict[int, TTEntry] = {}
        self.max_size = max_size
        self.current_age = 0
        self.hits = 0
        self.misses = 0
    
    def get(self, key: int) -> Optional[TTEntry]:
        entry = self.table.get(int(key))
        if entry:
            self.hits += 1
            return entry
        self.misses += 1
        return None
    
    def put(self, key: int, entry: TTEntry):
        entry.age = self.current_age
        self.table[int(key)] = entry
        
        # Clear old entries if table too large
        if len(self.table) > self.max_size:
            self._prune_old_entries()
    
    def _prune_old_entries(self):
        """Remove entries older than 2 ages"""
        cutoff_age = self.current_age - 2
        self.table = {k: v for k, v in self.table.items() if v.age >= cutoff_age}
    
    def new_search(self):
        """Call at start of new search to age entries"""
        self.current_age += 1
    
    def clear(self):
        self.table.clear()
        self.hits = 0
        self.misses = 0

# KILLER MOVES & HISTORY HEURISTIC
class KillerMoves:
    """Store killer moves for each depth (up to 2 per depth)"""
    def __init__(self, max_depth: int = MAX_DEPTH):
        self.killers = [[None, None] for _ in range(max_depth)]
    
    def add(self, depth: int, move: Tuple[int,int]):
        if depth >= len(self.killers):
            return
        if move != self.killers[depth][0]:
            self.killers[depth][1] = self.killers[depth][0]
            self.killers[depth][0] = move
    
    def get(self, depth: int) -> List[Tuple[int,int]]:
        if depth >= len(self.killers):
            return []
        return [m for m in self.killers[depth] if m is not None]
    
    def clear(self):
        for i in range(len(self.killers)):
            self.killers[i] = [None, None]

class HistoryTable:
    """History heuristic: track moves that caused cutoffs"""
    def __init__(self, board_size: int = 15):
        self.size = board_size
        self.scores = np.zeros((board_size, board_size), dtype=np.int32)
    
    def add(self, move: Tuple[int,int], depth: int):
        r, c = move
        self.scores[r, c] += depth * depth  # depth^2 weighting
    
    def get_score(self, move: Tuple[int,int]) -> int:
        r, c = move
        return int(self.scores[r, c])
    
    def clear(self):
        self.scores.fill(0)

# THREAT DETECTION (Caro-specific)
def detect_threats(board: Board, player: int) -> List[Tuple[int,int]]:
    """
    Detect forcing moves (threats):
    - 4-in-a-row (must block)
    - Open 3-in-a-row (high priority)
    - 3-in-a-row with one end blocked
    """
    threats = []
    dirs = [(0,1), (1,0), (1,1), (1,-1)]
    
    for r in range(board.size):
        for c in range(board.size):
            if board.grid[r,c] != 0:
                continue
            
            # Check if this empty square creates/blocks threats
            for dr, dc in dirs:
                # Count consecutive stones in both directions
                count_player = 0
                count_opp = 0
                
                # Check player's perspective
                for sign in [1, -1]:
                    rr, cc = r + sign*dr, c + sign*dc
                    while board.in_bounds(rr, cc) and board.grid[rr, cc] == player:
                        count_player += 1
                        rr += sign*dr
                        cc += sign*dc
                
                # Check opponent's perspective
                for sign in [1, -1]:
                    rr, cc = r + sign*dr, c + sign*dc
                    while board.in_bounds(rr, cc) and board.grid[rr, cc] == -player:
                        count_opp += 1
                        rr += sign*dr
                        cc += sign*dc
                
                # 4-in-a-row threat (immediate win/block)
                if count_player >= 4 or count_opp >= 4:
                    return [(r, c)]  # Return immediately, must respond
                
                # 3-in-a-row threat
                if count_player >= 3 or count_opp >= 3:
                    threats.append((r, c))
    
    return threats

# MOVE ORDERING
def order_moves(board: Board, moves: List[Tuple[int,int]], 
                tt_move: Optional[Tuple[int,int]],
                killer_moves: List[Tuple[int,int]],
                history: HistoryTable,
                player: int,
                policy_scores: Optional[Dict[Tuple[int,int], float]] = None,
                depth: int = 0) -> List[Tuple[int,int]]:
    """
    Advanced move ordering:
    1. TT move (PV)
    2. Threats (4/3-in-a-row)
    3. Policy network scores (if available)
    4. Killer moves
    5. History heuristic
    6. Proximity to existing stones

    for shallow depths (<3), skip expensive ordering to reduce overhead
    """
    if not moves:
        return []
    
    # For very shallow searches, just use simple ordering
    if depth <= 2 and not policy_scores:
        # Quick ordering: TT move first, then others
        if tt_move and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
        return moves
    
    # Priority 1: TT move
    if tt_move and tt_move in moves:
        moves.remove(tt_move)
        moves.insert(0, tt_move)
        return moves
    
    # Priority 2: Detect threats
    threats = detect_threats(board, player)
    if threats and threats[0] in moves:
        # Immediate threat found
        moves.remove(threats[0])
        moves.insert(0, threats[0])
        if len(threats) > 1:
            for t in threats[1:]:
                if t in moves:
                    moves.remove(t)
                    moves.insert(1, t)
        return moves
    
    # Score remaining moves
    scored_moves = []
    for mv in moves:
        score = 0.0
        r, c = mv
        
        # Policy network score (highest weight)
        if policy_scores and mv in policy_scores:
            score += policy_scores[mv] * 10000
        
        # Killer move bonus
        if mv in killer_moves:
            score += 5000
        
        # History heuristic
        score += history.get_score(mv)
        
        # Quick pattern evaluation
        board.grid[r,c] = player
        pattern_score = evaluate_pattern(board, player)
        board.grid[r,c] = 0
        score += pattern_score * 10
        
        scored_moves.append((score, mv))
    
    # Sort descending by score
    scored_moves.sort(reverse=True, key=lambda x: x[0])
    return [mv for _, mv in scored_moves]

# QUIESCENCE SEARCH
def quiescence_search(board: Board, alpha: float, beta: float, 
                      player: int, evaluate_fn) -> float:
    """
    Extend search for forcing moves (threats) to avoid horizon effect
    """
    stand_pat = evaluate_fn(board, player)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    
    # Only search threat moves
    threats = detect_threats(board, player)
    if not threats:
        return stand_pat
    
    for mv in threats[:3]:  # Limit to top 3 threats
        r, c = mv
        board.play(r, c, player)
        
        if board.is_win_from(r, c):
            score = MATE_SCORE
        else:
            score = -quiescence_search(board, -beta, -alpha, -player, evaluate_fn)
        
        board.undo()
        
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    
    return alpha


# PRINCIPAL VARIATION SEARCH (PVS)
def pvs(board: Board, depth: int, alpha: float, beta: float,
        player: int, tt: TranspositionTable, 
        killers: KillerMoves, history: HistoryTable,
        evaluate_fn, stats: dict, 
        policy_scores: Optional[Dict] = None,
        original_depth: int = None) -> float:
    """
    Principal Variation Search with all optimizations
    """
    if original_depth is None:
        original_depth = depth
        stats['original_depth'] = original_depth
    
    stats['nodes'] += 1
    
    # === TERMINAL CHECK ===
    if board.last_move is not None:
        lr, lc = board.last_move
        if board.is_win_from(lr, lc):
            return -(MATE_SCORE - board.move_count)
    
    # === DEPTH LIMIT ===
    if depth == 0:
        # Only use quiescence for deep searches (depth >= 4 originally)
        # Shallow searches don't benefit from quiescence overhead
        if stats.get('original_depth', 10) >= 4:
            return quiescence_search(board, alpha, beta, player, evaluate_fn)
        else:
            return evaluate_fn(board, player)
    
    # === TT LOOKUP ===
    key = int(board.zobrist_hash)
    tt_entry = tt.get(key)
    tt_move = None
    
    alpha_orig = alpha
    if tt_entry and tt_entry.depth >= depth:
        tt_move = tt_entry.best_move
        
        if tt_entry.flag == "EXACT":
            stats['tt_hits'] += 1
            return tt_entry.value
        elif tt_entry.flag == "LOWER":
            alpha = max(alpha, tt_entry.value)
        elif tt_entry.flag == "UPPER":
            beta = min(beta, tt_entry.value)
        
        if alpha >= beta:
            stats['tt_cutoffs'] += 1
            return tt_entry.value
    
    # === MOVE GENERATION & ORDERING ===
    moves = board.generate_candidates(radius=2)
    if not moves:
        return 0.0  # Draw
    
    killer_moves_list = killers.get(depth)
    moves = order_moves(board, moves, tt_move, killer_moves_list, 
                       history, player, policy_scores, depth)  # Pass depth
    
    # === MAIN SEARCH ===
    best_value = -INF
    best_move = None
    first_move = True
    moves_searched = 0
    
    for mv in moves:
        r, c = mv
        board.play(r, c, player)
        
        # Immediate win detection
        if board.is_win_from(r, c):
            val = MATE_SCORE - depth
            board.undo()
            
            # Store in TT
            tt.put(key, TTEntry(depth, val, "EXACT", mv))
            return val
        
        # === PVS Logic ===
        if first_move:
            # Full window search for PV node
            val = -pvs(board, depth-1, -beta, -alpha, -player, 
                      tt, killers, history, evaluate_fn, stats, policy_scores)
            first_move = False
        else:
            # === Late Move Reduction (LMR) ===
            reduction = 0
            if moves_searched >= 3 and depth >= 3:
                reduction = 1
                if moves_searched >= 6 and depth >= 5:
                    reduction = 2
            
            # Null window search
            val = -pvs(board, depth-1-reduction, -alpha-1, -alpha, -player,
                      tt, killers, history, evaluate_fn, stats, policy_scores)
            
            # Re-search if needed
            if alpha < val < beta:
                stats['re_searches'] += 1
                val = -pvs(board, depth-1, -beta, -val, -player,
                          tt, killers, history, evaluate_fn, stats, policy_scores)
        
        board.undo()
        moves_searched += 1
        
        # === Update best ===
        if val > best_value:
            best_value = val
            best_move = mv
        
        # === Alpha-Beta cutoff ===
        if val > alpha:
            alpha = val
        
        if alpha >= beta:
            stats['cutoffs'] += 1
            # Update killer & history
            if mv not in killer_moves_list:
                killers.add(depth, mv)
            history.add(mv, depth)
            break
    
    # === TT STORE ===
    if best_value <= alpha_orig:
        flag = "UPPER"
    elif best_value >= beta:
        flag = "LOWER"
    else:
        flag = "EXACT"
    
    tt.put(key, TTEntry(depth, best_value, flag, best_move))
    
    return best_value

# ITERATIVE DEEPENING WITH ASPIRATION WINDOWS
def get_best_move(board: Board, max_depth: int = 6, max_time: float = 2.0,
                  player: int = 1, evaluate_fn = evaluate_model,
                  policy_scores: Optional[Dict] = None,
                  verbose: bool = False) -> Tuple[Optional[Tuple[int,int]], float, dict]:
    """
    Main search function with iterative deepening and aspiration windows
    
    Returns:
        (best_move, best_value, stats)
    """
    start_time = time.perf_counter()
    
    # Initialize data structures
    tt = TranspositionTable()
    tt.new_search()
    killers = KillerMoves()
    history = HistoryTable(board.size)
    
    stats = {
        'nodes': 0,
        'cutoffs': 0,
        'tt_hits': 0,
        'tt_cutoffs': 0,
        're_searches': 0,
        'depth_reached': 0,
        'time_used': 0.0
    }
    
    best_move = None
    best_value = -INF
    
    # === ITERATIVE DEEPENING ===
    for depth in range(1, max_depth + 1):
        # Time check
        elapsed = time.perf_counter() - start_time
        if elapsed > max_time * 0.9:  # 90% time used
            break
        
        # === ASPIRATION WINDOWS ===
        if depth <= 3:
            # Full window for shallow depths
            alpha, beta = -INF, INF
        else:
            # Aspiration window around previous score
            window = 50
            alpha = best_value - window
            beta = best_value + window
        
        # Search with aspiration window
        local_best_move = None
        local_best_value = -INF
        
        moves = board.generate_candidates(radius=2)
        
        # Order moves using previous best
        if best_move and best_move in moves:
            moves.remove(best_move)
            moves.insert(0, best_move)
        
        for mv in moves:
            # Time check
            if time.perf_counter() - start_time > max_time:
                break
            
            r, c = mv
            board.play(r, c, player)
            
            # Immediate win
            if board.is_win_from(r, c):
                val = MATE_SCORE - depth
                board.undo()
                
                if verbose:
                    print(f"[Depth {depth}] WINNING MOVE FOUND: {mv}")
                
                stats['depth_reached'] = depth
                stats['time_used'] = time.perf_counter() - start_time
                return mv, val, stats
            
            # Search
            val = -pvs(board, depth-1, -beta, -alpha, -player,
                      tt, killers, history, evaluate_fn, stats, policy_scores)
            board.undo()
            
            # Aspiration window fail-high/low
            if val <= alpha or val >= beta:
                # Re-search with full window
                board.play(r, c, player)
                val = -pvs(board, depth-1, -INF, INF, -player,
                          tt, killers, history, evaluate_fn, stats, policy_scores)
                board.undo()
            
            if val > local_best_value:
                local_best_value = val
                local_best_move = mv
        
        # Update best from this depth
        if local_best_move is not None:
            best_move = local_best_move
            best_value = local_best_value
            stats['depth_reached'] = depth
        
        if verbose:
            elapsed = time.perf_counter() - start_time
            print(f"[Depth {depth}] Best: {best_move}, Value: {best_value:.1f}, "
                  f"Nodes: {stats['nodes']}, Time: {elapsed:.2f}s")
        
        # Early exit if mate found
        if abs(best_value) > MATE_SCORE - 100:
            break
    
    stats['time_used'] = time.perf_counter() - start_time
    
    if verbose:
        print(f"\n=== SEARCH COMPLETE ===")
        print(f"Best move: {best_move}")
        print(f"Value: {best_value:.1f}")
        print(f"Depth reached: {stats['depth_reached']}")
        print(f"Total nodes: {stats['nodes']}")
        print(f"Cutoffs: {stats['cutoffs']}")
        print(f"TT hits: {stats['tt_hits']}")
        print(f"Time: {stats['time_used']:.2f}s")
    
    return best_move, best_value, stats

# CONVENIENCE WRAPPER (backward compatibility)
def negamax(board: Board, depth: int, alpha: float, beta: float,
            player: int, tt: TranspositionTable, evaluate_fn) -> float:
    """Wrapper for backward compatibility"""
    killers = KillerMoves()
    history = HistoryTable(board.size)
    stats = {'nodes': 0, 'cutoffs': 0, 'tt_hits': 0, 'tt_cutoffs': 0, 're_searches': 0}
    return pvs(board, depth, alpha, beta, player, tt, killers, history, 
               evaluate_fn, stats)

