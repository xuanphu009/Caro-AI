"""
BENCHMARK - Compare performance with and without model
Measures: speed, accuracy, win rate
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from game import Board
from searchs import get_best_move
import time
import numpy as np

class Benchmark:
    def __init__(self, model_path="checkpoints/caro_best.pt"):
        self.model_path = model_path
        self.results = {}
    
    def benchmark_single_search(self, depth=3, use_model=False):
        """Benchmark single search"""
        board = Board()
        
        # Setup board with some moves
        positions = [
            (7, 7, 1), (8, 8, -1), (7, 8, 1), (9, 9, -1),
            (6, 6, 1), (10, 10, -1)
        ]
        for r, c, p in positions:
            board.play(r, c, p)
        
        times = []
        nodes_list = []
        
        for _ in range(3):  # 3 runs
            start = time.time()
            move, value, stats = get_best_move(
                board=board.copy() if hasattr(board, 'copy') else board,
                max_depth=depth,
                max_time=10.0,
                player=1,
                use_model=use_model,
                model_path=self.model_path if use_model else None
            )
            elapsed = time.time() - start
            
            times.append(elapsed)
            nodes_list.append(stats['nodes'])
        
        return {
            'avg_time': np.mean(times),
            'avg_nodes': np.mean(nodes_list),
            'throughput': np.mean(nodes_list) / np.mean(times)
        }
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "="*70)
        print("PERFORMANCE BENCHMARK - Heuristic vs Model")
        print("="*70)
        
        depths = [2, 3, 4]
        
        for depth in depths:
            print(f"\n{'='*70}")
            print(f"DEPTH {depth}")
            print(f"{'='*70}")
            
            # Heuristic only
            print(f"\n[1/2] Benchmarking heuristic-only search...")
            heur_results = self.benchmark_single_search(depth=depth, use_model=False)
            print(f"✅ Heuristic Results:")
            print(f"   Avg time:      {heur_results['avg_time']:.2f}s")
            print(f"   Avg nodes:     {heur_results['avg_nodes']:.0f}")
            print(f"   Throughput:    {heur_results['throughput']:.0f} nodes/sec")
            
            # Model-fused
            if os.path.exists(self.model_path):
                print(f"\n[2/2] Benchmarking model-fused search...")
                try:
                    model_results = self.benchmark_single_search(depth=depth, use_model=True)
                    print(f"✅ Model Results:")
                    print(f"   Avg time:      {model_results['avg_time']:.2f}s")
                    print(f"   Avg nodes:     {model_results['avg_nodes']:.0f}")
                    print(f"   Throughput:    {model_results['throughput']:.0f} nodes/sec")
                    
                    # Comparison
                    print(f"\n📊 Comparison:")
                    time_ratio = heur_results['avg_time'] / model_results['avg_time']
                    node_ratio = heur_results['avg_nodes'] / model_results['avg_nodes']
                    
                    print(f"   Speed ratio:   {time_ratio:.2f}x {'(model faster)' if time_ratio > 1 else '(heuristic faster)'}")
                    print(f"   Nodes ratio:   {node_ratio:.2f}x {'(model fewer)' if node_ratio > 1 else '(heuristic fewer)'}")
                    
                    self.results[depth] = {
                        'heuristic': heur_results,
                        'model': model_results,
                        'speedup': time_ratio,
                        'efficiency': node_ratio
                    }
                except Exception as e:
                    print(f"❌ Model benchmark failed: {e}")
                    print("   Model may not be trained yet.")
            else:
                print(f"\n⚠️ Model not found: {self.model_path}")
                print("   Train model first: python src/train_cnn.py")
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary"""
        if not self.results:
            return
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        for depth, data in self.results.items():
            print(f"\nDepth {depth}:")
            print(f"  Heuristic: {data['heuristic']['avg_time']:.2f}s ({data['heuristic']['avg_nodes']:.0f} nodes)")
            print(f"  Model:     {data['model']['avg_time']:.2f}s ({data['model']['avg_nodes']:.0f} nodes)")
            print(f"  Model is {data['speedup']:.2f}x {'faster' if data['speedup'] > 1 else 'slower'}")
            print(f"  Model explores {data['efficiency']:.2f}x {'fewer' if data['efficiency'] > 1 else 'more'} nodes")


def test_move_quality():
    """Compare move quality"""
    print("\n" + "="*70)
    print("MOVE QUALITY COMPARISON")
    print("="*70)
    
    board = Board()
    positions = [(7, 7, 1), (8, 8, -1)]
    for r, c, p in positions:
        board.play(r, c, p)
    
    print(f"\nBoard setup: {positions}")
    
    # Get moves
    print(f"\n[1/2] Heuristic-only move...")
    move_h, value_h, stats_h = get_best_move(
        board=board,
        max_depth=4,
        max_time=5.0,
        player=1,
        use_model=False
    )
    print(f"✅ Heuristic: {move_h}, Value: {value_h:.2f}")
    print(f"   Nodes: {stats_h['nodes']}, Time: {stats_h['time_used']:.2f}s")
    
    if os.path.exists("checkpoints/caro_best.pt"):
        print(f"\n[2/2] Model-fused move...")
        try:
            move_m, value_m, stats_m = get_best_move(
                board=board,
                max_depth=4,
                max_time=5.0,
                player=1,
                use_model=True,
                model_path="checkpoints/caro_best.pt"
            )
            print(f"✅ Model: {move_m}, Value: {value_m:.2f}")
            print(f"   Nodes: {stats_m['nodes']}, Time: {stats_m['time_used']:.2f}s")
            
            # Analysis
            print(f"\n📊 Analysis:")
            same_move = move_h == move_m
            print(f"   Same move: {'✅ YES' if same_move else '❌ NO'}")
            print(f"   Value difference: {abs(value_m - value_h):.2f}")
            print(f"   Model {'explored' if stats_m['nodes'] < stats_h['nodes'] else 'searched'} "
                  f"{abs(stats_m['nodes'] - stats_h['nodes'])} {'fewer' if stats_m['nodes'] < stats_h['nodes'] else 'more'} nodes")
        except Exception as e:
            print(f"❌ Model comparison failed: {e}")
    else:
        print(f"\n⚠️ Model not available")


def main():
    print("\n" + "╔"+"="*68+"╗")
    print("║" + " "*20 + "CARO AI BENCHMARK" + " "*32 + "║")
    print("╚"+"="*68+"╝")
    
    benchmark = Benchmark()
    
    # Run benchmarks
    benchmark.run_benchmarks()
    
    # Test move quality
    test_move_quality()
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    print("\nInterpretation:")
    print("• If model is slower but explores fewer nodes: model provides better evaluation")
    print("• If both choose same move: model agrees with heuristic")
    print("• If model chooses different move: model found better strategy")
    print("="*70)


if __name__ == "__main__":
    main()