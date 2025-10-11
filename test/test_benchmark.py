"""
Benchmark Test: So sánh Old Search vs New Optimized Search
Đo: thời gian, nodes searched, depth reached, move quality
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import time
import numpy as np
from typing import List, Tuple, Dict
from game import Board
from evaluate import evaluate_pattern

# ============================================================================
# OLD SEARCH IMPLEMENTATION (from your original code)
# ============================================================================
class OldTTEntry:
    def __init__(self, depth: int, value: float, flag: str, best_move):
        self.depth = depth
        self.value = value
        self.flag = flag
        self.best_move = best_move

class OldTranspositionTable:
    def __init__(self):
        self.table = {}
    
    def get(self, key: int):
        return self.table.get(int(key), None)
    
    def put(self, key: int, entry):
        self.table[int(key)] = entry

def old_negamax(board: Board, depth: int, alpha: float, beta: float,
                player: int, tt: OldTranspositionTable, evaluate_fn, stats: dict) -> float:
    """Old negamax implementation (original code)"""
    stats['nodes'] += 1
    
    # Check terminal by last move
    if board.last_move is not None:
        lr, lc = board.last_move
        if board.is_win_from(lr, lc):
            return -1e9 + 1
    
    if depth == 0:
        return evaluate_fn(board, player)
    
    key = int(board.zobrist_hash)
    tt_entry = tt.get(key)
    if tt_entry and tt_entry.depth >= depth:
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
    
    best_value = -1e9
    best_move = None
    
    # Simple move ordering (just pattern evaluation)
    moves = board.generate_candidates(radius=2)
    scored_moves = []
    for mv in moves:
        r,c = mv
        board.grid[r,c] = player
        s = evaluate_pattern(board, player)
        board.grid[r,c] = 0
        scored_moves.append((s, mv))
    
    scored_moves.sort(reverse=True, key=lambda x: x[0])
    ordered_moves = [mv for (_, mv) in scored_moves]
    
    for mv in ordered_moves:
        r,c = mv
        board.play(r,c,player)
        val = -old_negamax(board, depth-1, -beta, -alpha, -player, tt, evaluate_fn, stats)
        board.undo()
        
        if val > best_value:
            best_value = val
            best_move = mv
        alpha = max(alpha, val)
        if alpha >= beta:
            stats['cutoffs'] += 1
            break
    
    # Save to TT
    if best_value <= alpha:
        flag = "UPPER"
    elif best_value >= beta:
        flag = "LOWER"
    else:
        flag = "EXACT"
    tt.put(key, OldTTEntry(depth=depth, value=best_value, flag=flag, best_move=best_move))
    return best_value

def old_get_best_move(board: Board, max_depth: int = 4, max_time: float = None, 
                      player: int = 1, evaluate_fn = evaluate_pattern) -> Tuple:
    """Old search with iterative deepening"""
    start = time.perf_counter()
    tt = OldTranspositionTable()
    best_move = None
    best_value = -1e9
    
    stats = {
        'nodes': 0,
        'cutoffs': 0,
        'tt_hits': 0,
        'tt_cutoffs': 0,
        'depth_reached': 0
    }
    
    for d in range(1, max_depth+1):
        if max_time is not None and (time.perf_counter() - start) > max_time:
            break
        
        local_best_move = None
        local_best_value = -1e9
        
        moves = board.generate_candidates(radius=2)
        if best_move in moves:
            moves.remove(best_move)
            moves.insert(0, best_move)
        
        for mv in moves:
            if max_time is not None and (time.perf_counter() - start) > max_time:
                break
            r,c = mv
            board.play(r,c, player)
            
            if board.is_win_from(r,c):
                val = 100000 - d
            else:
                val = -old_negamax(board, d-1, -1e9, 1e9, -player, tt, evaluate_fn, stats)
            board.undo()
            
            if val > local_best_value:
                local_best_value = val
                local_best_move = mv
        
        if local_best_move is not None:
            best_move = local_best_move
            best_value = local_best_value
            stats['depth_reached'] = d
    
    stats['time_used'] = time.perf_counter() - start
    return best_move, best_value, stats

# ============================================================================
# NEW SEARCH IMPLEMENTATION (import from optimized version)
# ============================================================================
try:
    from searchs import get_best_move as new_get_best_move
    NEW_AVAILABLE = True
except ImportError:
    print("⚠️  New searchs.py not found! Please save the optimized version first.")
    NEW_AVAILABLE = False

# ============================================================================
# TEST POSITIONS
# ============================================================================
def create_test_positions() -> List[Tuple[Board, str, int]]:
    """
    Create diverse test positions:
    - Opening (empty board)
    - Mid-game (10-15 moves)
    - Tactical (threat positions)
    - Endgame (winning in few moves)
    """
    positions = []
    
    # Position 1: Opening (center bias)
    board1 = Board(15)
    board1.play(7, 7, 1)
    board1.play(7, 8, -1)
    board1.play(8, 7, 1)
    positions.append((board1.copy() if hasattr(board1, 'copy') else board1, "Opening", 1))
    
    # Position 2: Mid-game (complex)
    board2 = Board(15)
    moves2 = [(7,7,1), (7,8,-1), (8,8,1), (8,7,-1), (9,9,1), (6,6,-1), 
              (6,7,1), (6,8,-1), (7,9,1), (8,9,-1)]
    for r,c,p in moves2:
        board2.play(r,c,p)
    positions.append((board2, "Mid-game", 1))
    
    # Position 3: Tactical (player has 3-in-a-row, must win)
    board3 = Board(15)
    # Create horizontal 3-in-a-row for player 1
    board3.play(7, 5, 1)
    board3.play(6, 5, -1)
    board3.play(7, 6, 1)
    board3.play(6, 6, -1)
    board3.play(7, 7, 1)
    # Now (7,4) or (7,8) wins for player 1
    positions.append((board3, "Tactical (3-in-row)", 1))
    
    # Position 4: Defensive (opponent has 3-in-a-row, must block)
    board4 = Board(15)
    board4.play(8, 5, -1)
    board4.play(7, 5, 1)
    board4.play(8, 6, -1)
    board4.play(7, 6, 1)
    board4.play(8, 7, -1)
    # Player 1 must block at (8,4) or (8,8)
    positions.append((board4, "Defensive (must block)", 1))
    
    # Position 5: Complex endgame
    board5 = Board(15)
    moves5 = [(7,7,1), (7,8,-1), (8,7,1), (8,8,-1), (9,7,1), (9,8,-1),
              (6,7,1), (6,8,-1), (7,6,1), (8,6,-1), (9,6,1), (10,6,-1),
              (5,7,1), (5,8,-1), (7,9,1)]
    for r,c,p in moves5:
        board5.play(r,c,p)
    positions.append((board5, "Endgame", -1))
    
    return positions

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================
def run_single_benchmark(board: Board, position_name: str, player: int,
                        max_depth: int, max_time: float) -> Dict:
    """Run benchmark for one position with both old and new search"""
    
    print(f"\n{'='*60}")
    print(f"Position: {position_name} | Player: {player}")
    print(f"Max Depth: {max_depth} | Max Time: {max_time}s")
    print(f"{'='*60}")
    
    results = {}
    
    # === OLD SEARCH ===
    print("\n🔵 OLD SEARCH:")
    board_copy = Board(board.size)
    board_copy.grid = board.grid.copy()
    board_copy.move_stack = board.move_stack.copy()
    board_copy.last_move = board.last_move
    board_copy.zobrist_hash = board.zobrist_hash
    board_copy.move_count = board.move_count
    
    try:
        old_move, old_value, old_stats = old_get_best_move(
            board_copy, 
            max_depth=max_depth,
            max_time=max_time,
            player=player
        )
        
        results['old'] = {
            'move': old_move,
            'value': old_value,
            'time': old_stats['time_used'],
            'nodes': old_stats['nodes'],
            'depth': old_stats['depth_reached'],
            'cutoffs': old_stats['cutoffs'],
            'tt_hits': old_stats['tt_hits'],
            'success': True
        }
        
        print(f"  Move: {old_move}")
        print(f"  Value: {old_value:.1f}")
        print(f"  Time: {old_stats['time_used']:.3f}s")
        print(f"  Nodes: {old_stats['nodes']:,}")
        print(f"  Depth: {old_stats['depth_reached']}")
        print(f"  Cutoffs: {old_stats['cutoffs']:,}")
        print(f"  TT Hits: {old_stats['tt_hits']:,}")
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results['old'] = {'success': False, 'error': str(e)}
    
    # === NEW SEARCH ===
    if NEW_AVAILABLE:
        print("\n🟢 NEW OPTIMIZED SEARCH:")
        board_copy = Board(board.size)
        board_copy.grid = board.grid.copy()
        board_copy.move_stack = board.move_stack.copy()
        board_copy.last_move = board.last_move
        board_copy.zobrist_hash = board.zobrist_hash
        board_copy.move_count = board.move_count
        
        try:
            new_move, new_value, new_stats = new_get_best_move(
                board_copy,
                max_depth=max_depth,
                max_time=max_time,
                player=player,
                verbose=False
            )
            
            results['new'] = {
                'move': new_move,
                'value': new_value,
                'time': new_stats['time_used'],
                'nodes': new_stats['nodes'],
                'depth': new_stats['depth_reached'],
                'cutoffs': new_stats['cutoffs'],
                'tt_hits': new_stats['tt_hits'],
                're_searches': new_stats.get('re_searches', 0),
                'success': True
            }
            
            print(f"  Move: {new_move}")
            print(f"  Value: {new_value:.1f}")
            print(f"  Time: {new_stats['time_used']:.3f}s")
            print(f"  Nodes: {new_stats['nodes']:,}")
            print(f"  Depth: {new_stats['depth_reached']}")
            print(f"  Cutoffs: {new_stats['cutoffs']:,}")
            print(f"  TT Hits: {new_stats['tt_hits']:,}")
            print(f"  Re-searches: {new_stats.get('re_searches', 0):,}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results['new'] = {'success': False, 'error': str(e)}
    
    # === COMPARISON ===
    if results.get('old', {}).get('success') and results.get('new', {}).get('success'):
        print(f"\n📊 COMPARISON:")
        
        speedup = results['old']['time'] / results['new']['time'] if results['new']['time'] > 0 else 0
        node_reduction = (1 - results['new']['nodes'] / results['old']['nodes']) * 100 if results['old']['nodes'] > 0 else 0
        depth_gain = results['new']['depth'] - results['old']['depth']
        
        print(f"  ⚡ Speedup: {speedup:.2f}x")
        print(f"  📉 Node Reduction: {node_reduction:.1f}%")
        print(f"  📈 Depth Gain: +{depth_gain}")
        print(f"  🎯 Same Move: {'✅ Yes' if results['old']['move'] == results['new']['move'] else '❌ No'}")
        
        results['comparison'] = {
            'speedup': speedup,
            'node_reduction': node_reduction,
            'depth_gain': depth_gain,
            'same_move': results['old']['move'] == results['new']['move']
        }
    
    return results

# ============================================================================
# MAIN BENCHMARK
# ============================================================================
"""Run full benchmark suite"""
print("="*60)
print("🏁 BENCHMARK: Old Search vs New Optimized Search")
print("="*60)

positions = create_test_positions()

# Test configurations
configs = [
    {'max_depth': 4, 'max_time': 1.0, 'name': 'Fast (depth=4, 1s)'},
    {'max_depth': 6, 'max_time': 2.0, 'name': 'Normal (depth=6, 2s)'},
    {'max_depth': 8, 'max_time': 5.0, 'name': 'Deep (depth=8, 5s)'},
]

all_results = []

for config in configs:
    print(f"\n\n{'#'*60}")
    print(f"CONFIG: {config['name']}")
    print(f"{'#'*60}")
    
    for i, (board, pos_name, player) in enumerate(positions, 1):
        result = run_single_benchmark(
            board, 
            f"{i}. {pos_name}", 
            player,
            config['max_depth'],
            config['max_time']
        )
        all_results.append({
            'config': config['name'],
            'position': pos_name,
            'result': result
        })

# === SUMMARY ===
print("\n\n" + "="*60)
print("📊 OVERALL SUMMARY")
print("="*60)

if NEW_AVAILABLE:
    speedups = []
    node_reductions = []
    depth_gains = []
    
    for r in all_results:
        if 'comparison' in r['result']:
            comp = r['result']['comparison']
            speedups.append(comp['speedup'])
            node_reductions.append(comp['node_reduction'])
            depth_gains.append(comp['depth_gain'])
    
    if speedups:
        print(f"\n🚀 Average Speedup: {np.mean(speedups):.2f}x (min: {np.min(speedups):.2f}x, max: {np.max(speedups):.2f}x)")
        print(f"📉 Average Node Reduction: {np.mean(node_reductions):.1f}%")
        print(f"📈 Average Depth Gain: +{np.mean(depth_gains):.1f}")
        print(f"\n🎯 Move Agreement: {sum(1 for r in all_results if r['result'].get('comparison', {}).get('same_move', False))}/{len(all_results)} positions")

print("\n✅ Benchmark complete!")