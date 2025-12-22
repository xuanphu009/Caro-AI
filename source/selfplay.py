
import os
import sys
import json
import time
import random
import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source.search import CNN_GomokuAI, GomokuAI

# OPENING BOOK - Professional patterns
OPENING_PATTERNS = {
    # Center openings (balanced)
    "center": [(7, 7)],
    
    # Off-center openings (more aggressive)
    "diagonal": [(7, 7), (6, 6), (8, 8)],
    
    # Professional patterns
    "soosyrv": [(7, 7), (7, 6), (6, 7)],  # Classic opening
    "flower": [(7, 7), (6, 7), (7, 8)],    # Balanced
    "sword": [(7, 7), (6, 6), (8, 6)],     # Aggressive
}


def get_opening_move(move_count: int, pattern: str = None) -> Optional[Tuple[int, int]]:
    """
    Get opening book move based on game phase
    Returns None if out of opening book
    """
    if move_count == 0:
        # First move: slightly randomized center
        return (random.randint(6, 8), random.randint(6, 8))
    
    if move_count <= 3 and pattern:
        patterns = OPENING_PATTERNS.get(pattern, OPENING_PATTERNS["center"])
        if move_count < len(patterns):
            base_r, base_c = patterns[move_count]
            # Add small random offset for diversity
            r = base_r + random.randint(-1, 1)
            c = base_c + random.randint(-1, 1)
            return (max(4, min(10, r)), max(4, min(10, c)))
    
    return None

# ADVANCED SELF-PLAY ENGINE
class AdvancedSelfPlay:
    """
    High-quality self-play game generator with CNN guidance
    """
    
    def __init__(
        self,
        save_dir: str = "data/selfplay",
        model_path: str = "checkpoints/caro_best.pt",
        max_depth: int = 4,
        use_cnn: bool = True,
        temperature: float = 0.5,
        quality_threshold: float = 0.5,
        opening_diversity: bool = True
    ):
        self.save_dir = save_dir
        self.max_depth = max_depth
        self.use_cnn = use_cnn
        self.temperature = temperature
        self.quality_threshold = quality_threshold
        self.opening_diversity = opening_diversity
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_games': 0,
            'total_moves': 0,
            'wins_p1': 0,
            'wins_p2': 0,
            'draws': 0,
            'avg_game_length': 0,
            'quality_filtered': 0
        }
        
        # Initialize AI engine
        if use_cnn and os.path.exists(model_path):
            print(f"🧠 Loading CNN model: {model_path}")
            self.ai_class = CNN_GomokuAI
            self.model_path = model_path
        else:
            print("⚠️  CNN model not found, using heuristic-only AI")
            self.ai_class = GomokuAI
            self.model_path = None
    
    def _adaptive_depth(self, move_count: int, total_empty: int) -> int:
        """
        Adaptive search depth based on game phase
        - Opening (0-10 moves): Shallow depth (save time)
        - Midgame (10-100 moves): Normal depth
        - Endgame (>100 moves): Deep depth (critical)
        """
        if move_count < 10:
            return max(2, self.max_depth - 2)  # Opening: fast play
        elif move_count < 100:
            return self.max_depth  # Midgame: normal
        elif total_empty < 30:
            return min(6, self.max_depth + 2)  # Endgame: deep search
        else:
            return self.max_depth
    
    def _temperature_sampling(
        self,
        env: GomokuAI,
        temperature: float = 0.5
    ) -> Tuple[int, int]:
        """
        Temperature-based move sampling from top candidates
        Higher temperature = more exploration
        Lower temperature = more exploitation (stronger play)
        """
        if not env.nextBound:
            return (-1, -1)
        
        # Get valid moves with scores
        candidates = [
            (pos, score) 
            for pos, score in env.nextBound.items() 
            if env.isValid(pos[0], pos[1])
        ]
        
        if not candidates:
            return (-1, -1)
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Temperature = 0: Always pick best move
        if temperature < 0.01:
            return candidates[0][0]
        
        # Take top-k candidates
        top_k = min(10, len(candidates))
        top_candidates = candidates[:top_k]
        
        # Apply temperature to scores
        positions = [pos for pos, _ in top_candidates]
        scores = np.array([score for _, score in top_candidates], dtype=np.float64)
        
        # Subtract max for numerical stability
        scores = scores - scores.max()
        
        # Softmax with temperature (numerically stable)
        exp_scores = np.exp(scores / temperature)
        
        # Handle edge case where all exp_scores are 0 or inf
        if not np.isfinite(exp_scores).all() or exp_scores.sum() == 0:
            # Fallback: uniform sampling from top-3
            return random.choice(positions[:min(3, len(positions))])
        
        probs = exp_scores / exp_scores.sum()
        
        # Final safety check
        if not np.isfinite(probs).all():
            return random.choice(positions[:min(3, len(positions))])
        
        # Sample based on probabilities
        idx = np.random.choice(len(positions), p=probs)
        return positions[idx]
    
    def _evaluate_move_quality(self, env: GomokuAI, move: Tuple[int, int]) -> float:
        """
        Evaluate move quality (0-1 scale)
        Used for filtering low-quality games
        """
        if move not in env.nextBound:
            return 0.0
        
        move_score = env.nextBound[move]
        
        # Get max score in current position
        if env.nextBound:
            max_score = max(env.nextBound.values())
            if max_score > 0:
                return move_score / max_score
        
        return 0.5
    
    def _is_critical_position(self, env: GomokuAI) -> bool:
        """
        Detect if current position is critical (needs deeper search)
        - Immediate threats
        - Forcing sequences
        - Complex tactical positions
        """
        if not env.nextBound:
            return False
        
        # Check for high-value moves (threats)
        max_score = max(env.nextBound.values())
        return abs(max_score) > 50000  # Live-4 or better
    
    def play_single_game(
        self,
        game_id: int,
        opening_pattern: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Play one high-quality game
        Returns game data or None if filtered
        """
        # Initialize AI with CNN support
        if self.ai_class == CNN_GomokuAI:
            env = CNN_GomokuAI(
                model_path=self.model_path,
                use_hybrid=True
            )
        else:
            env = GomokuAI(depth=self.max_depth)
        
        moves: List[List[int]] = []
        move_qualities: List[float] = []
        
        # Random first player
        first_player = random.choice([1, -1])
        env.turn = first_player
        
        # Opening phase
        if self.opening_diversity and opening_pattern:
            opening_move = get_opening_move(0, opening_pattern)
            if opening_move:
                r, c = opening_move
                moves.append([r, c])
                env.setState(r, c, env.turn)
                env.updateBound(r, c, env.nextBound)
                env.turn *= -1
        
        # Main game loop
        move_count = len(moves)
        
        while move_count < 225:
            # Adaptive depth
            current_depth = self._adaptive_depth(move_count, env.emptyCells)
            env.depth = current_depth
            
            # Critical position detection
            if self._is_critical_position(env):
                current_depth = min(6, current_depth + 1)
                env.depth = current_depth
            
            # Ensure valid bound
            if not env.nextBound:
                if moves:
                    last_r, last_c = moves[-1]
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = last_r + dr, last_c + dc
                            if 0 <= nr < 15 and 0 <= nc < 15:
                                env.updateBound(nr, nc, env.nextBound)
            
            if not env.nextBound:
                break
            
            # Run alpha-beta search
            try:
                env.alphaBetaPruning(
                    env.depth,
                    env.boardValue,
                    env.nextBound,
                    -float('inf'),
                    float('inf'),
                    env.turn == 1
                )
            except Exception as e:
                print(f"⚠️  Search failed at move {move_count}: {e}")
                break
            
            # Temperature-based move selection
            if move_count < 5:
                # Opening: more diversity
                temp = self.temperature * 1.5
            elif move_count > 150:
                # Endgame: stronger play
                temp = self.temperature * 0.5
            else:
                temp = self.temperature
            
            move = self._temperature_sampling(env, temperature=temp)
            
            if move == (-1, -1) or not env.isValid(move[0], move[1]):
                break
            
            r, c = move
            
            # Evaluate move quality
            quality = self._evaluate_move_quality(env, move)
            move_qualities.append(quality)
            
            # Make move
            moves.append([r, c])
            env.setState(r, c, env.turn)
            env.updateBound(r, c, env.nextBound)
            
            # Check win
            if env.isFive(r, c, env.turn):
                winner = env.turn
                self.stats['wins_p1' if winner == 1 else 'wins_p2'] += 1
                
                # Quality check
                avg_quality = np.mean(move_qualities) if move_qualities else 0.5
                if avg_quality < self.quality_threshold:
                    self.stats['quality_filtered'] += 1
                    return None
                
                return self._create_game_record(
                    game_id, moves, winner, first_player,
                    current_depth, avg_quality
                )
            
            env.turn *= -1
            move_count += 1
            
            # Clear caches periodically
            if move_count % 50 == 0:
                env.clearCaches()
        
        # Draw
        self.stats['draws'] += 1
        avg_quality = np.mean(move_qualities) if move_qualities else 0.5
        
        # Filter low-quality draws
        if avg_quality < self.quality_threshold * 0.8:
            self.stats['quality_filtered'] += 1
            return None
        
        return self._create_game_record(
            game_id, moves, 0, first_player,
            current_depth, avg_quality
        )
    
    def _create_game_record(
        self,
        game_id: int,
        moves: List[List[int]],
        winner: int,
        first_player: int,
        max_depth: int,
        avg_quality: float
    ) -> Dict:
        """Create game record with metadata"""
        return {
            "moves": moves,
            "winner": winner,
            "first_player": first_player,
            "total_moves": len(moves),
            "metadata": {
                "game_id": game_id,
                "max_depth": max_depth,
                "avg_quality": round(avg_quality, 3),
                "strategy": "cnn_hybrid" if self.use_cnn else "heuristic",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def generate_games(
        self,
        n_games: int,
        workers: int = 1,
        show_progress: bool = True
    ):
        """
        Generate multiple high-quality games
        """
        print("\n" + "="*60)
        print("🎮 STARTING HIGH-QUALITY GAME GENERATION")
        print("="*60)
        print(f"Target games: {n_games}")
        print(f"Model: {'CNN-Hybrid' if self.use_cnn else 'Heuristic'}")
        print(f"Max depth: {self.max_depth}")
        print(f"Temperature: {self.temperature}")
        print(f"Quality threshold: {self.quality_threshold}")
        print(f"Workers: {workers}")
        print("="*60 + "\n")
        
        generated = 0
        attempted = 0
        
        # Opening patterns for diversity
        patterns = list(OPENING_PATTERNS.keys())
        
        if workers > 1:
            # Parallel generation
            print("⚠️  Parallel mode not yet implemented, using sequential")
            workers = 1
        
        # Sequential generation with progress bar
        pbar = tqdm(total=n_games, desc="Generating games") if show_progress else None
        
        while generated < n_games:
            attempted += 1
            
            # Random opening pattern
            pattern = random.choice(patterns) if self.opening_diversity else None
            
            # Play game
            game = self.play_single_game(generated, pattern)
            
            if game is not None:
                # Save game
                fname = f"selfplay_game_{generated:06d}.json"
                path = os.path.join(self.save_dir, fname)
                
                with open(path, "w") as f:
                    json.dump(game, f, indent=2)
                
                generated += 1
                self.stats['total_games'] += 1
                self.stats['total_moves'] += game['total_moves']
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'quality': f"{game['metadata']['avg_quality']:.2f}",
                        'moves': game['total_moves'],
                        'winner': game['winner']
                    })
            
            # Safety: prevent infinite loop
            if attempted > n_games * 3:
                print(f"\n⚠️  Quality threshold too high, stopping after {generated} games")
                break
        
        if pbar:
            pbar.close()
        
        self._print_statistics()
    
    def _print_statistics(self):
        """Print generation statistics"""
        print("\n" + "="*60)
        print("📊 GENERATION STATISTICS")
        print("="*60)
        
        total = self.stats['total_games']
        if total == 0:
            print("No games generated")
            return
        
        print(f"Total games: {total}")
        print(f"Total moves: {self.stats['total_moves']}")
        print(f"Avg moves/game: {self.stats['total_moves']/total:.1f}")
        print(f"\nResults:")
        print(f"  Player 1 wins: {self.stats['wins_p1']:4d} ({self.stats['wins_p1']/total*100:5.1f}%)")
        print(f"  Player 2 wins: {self.stats['wins_p2']:4d} ({self.stats['wins_p2']/total*100:5.1f}%)")
        print(f"  Draws:         {self.stats['draws']:4d} ({self.stats['draws']/total*100:5.1f}%)")
        print(f"\nQuality filtered: {self.stats['quality_filtered']}")
        print("="*60)

# CLI INTERFACE
def main():
    parser = argparse.ArgumentParser(
        description="Advanced Self-Play Game Generation for Caro AI"
    )
    
    parser.add_argument(
        "--n_games",
        type=int,
        default=100,
        help="Number of games to generate"
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/selfplay",
        help="Directory to save games"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/caro_best.pt",
        help="Path to CNN model checkpoint"
    )
    
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Base search depth (adaptive in-game)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (0=greedy, 1=random)"
    )
    
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.5,
        help="Minimum average move quality (0-1)"
    )
    
    parser.add_argument(
        "--no_cnn",
        action="store_true",
        help="Disable CNN model (heuristic only)"
    )
    
    parser.add_argument(
        "--no_opening_diversity",
        action="store_true",
        help="Disable opening book diversity"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (experimental)"
    )
    
    args = parser.parse_args()
    
    # Initialize self-play engine
    engine = AdvancedSelfPlay(
        save_dir=args.save_dir,
        model_path=args.model,
        max_depth=args.depth,
        use_cnn=not args.no_cnn,
        temperature=args.temperature,
        quality_threshold=args.quality_threshold,
        opening_diversity=not args.no_opening_diversity
    )
    
    # Generate games
    engine.generate_games(
        n_games=args.n_games,
        workers=args.workers,
        show_progress=True
    )
    
    print("\n✅ Generation complete!")
    print(f"💾 Games saved to: {args.save_dir}")


if __name__ == "__main__":
    main()