"""
Caro Game - Play against AI with optional CNN model
COMPLETE REWRITE - Fixed threading and event handling
"""

import pygame
import sys
import os
import argparse
import threading
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from game import Board
from searchs import get_best_move

SAVE_DIR = Path("data/run_game")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Constants
BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 20
WIDTH = HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)


class CaroGame:
    def __init__(self, use_model=False, model_path=None, ai_depth=3, ai_time=2.0):
        """Initialize game"""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Caro - Play against AI")
        
        self.board_logic = Board(BOARD_SIZE)
        self.turn = 1  # 1 = player, -1 = AI
        self.ai_depth = ai_depth
        self.ai_time = ai_time
        self.use_model = use_model
        self.model_path = model_path
        
        # Game state
        self.game_over = False
        self.winner = None
        self.message = "Your turn (X) - Click to place stone"
        
        # Save
        self.move_history = []

        # Threading
        self.ai_thread = None
        self.ai_result = None
        self.ai_running = False
        
        # Check model
        if use_model and model_path:
            if os.path.exists(model_path):
                print(f"Model found: {model_path}")
                self.use_model = True
            else:
                print(f"Model not found: {model_path}, using heuristic")
                self.use_model = False
    
    def draw_board(self):
        """Draw board and pieces"""
        self.screen.fill(WHITE)
        
        # Draw grid
        for i in range(BOARD_SIZE):
            # Horizontal lines
            pygame.draw.line(
                self.screen, GRAY,
                (MARGIN, MARGIN + i * CELL_SIZE),
                (WIDTH - MARGIN, MARGIN + i * CELL_SIZE)
            )
            # Vertical lines
            pygame.draw.line(
                self.screen, GRAY,
                (MARGIN + i * CELL_SIZE, MARGIN),
                (MARGIN + i * CELL_SIZE, HEIGHT - MARGIN)
            )
        
        # Draw stones
        font = pygame.font.SysFont("consolas", 24, bold=True)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = int(self.board_logic.grid[r, c])
                if val != 0:
                    text = font.render("X" if val == 1 else "O", True, BLACK)
                    self.screen.blit(
                        text,
                        (MARGIN + c * CELL_SIZE + 10, MARGIN + r * CELL_SIZE + 5)
                    )
        
        # Draw message
        msg_font = pygame.font.SysFont("consolas", 16)
        msg_text = msg_font.render(self.message, True, BLACK)
        self.screen.blit(msg_text, (10, HEIGHT - 30))
        
        # Draw AI thinking indicator
        if self.ai_running:
            think_text = msg_font.render("AI thinking...", True, LIGHT_BLUE)
            self.screen.blit(think_text, (WIDTH - 150, HEIGHT - 30))
    
    def handle_player_move(self, pos):
        """Handle player click"""
        if self.game_over:
            return
        if self.turn != 1:
            return
        if self.ai_running:
            print("AI still thinking, wait...")
            return
        
        x, y = pos
        col = (x - MARGIN) // CELL_SIZE
        row = (y - MARGIN) // CELL_SIZE
        
        # Validate
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return
        if self.board_logic.grid[row, col] != 0:
            return
        
        # Play
        self.board_logic.play(row, col, 1)
        print(f"Player move: ({row}, {col})")
        # Save
        self.move_history.append((row, col))
        
        # Check win
        if self.board_logic.is_win_from(row, col):
            self.game_over = True
            self.winner = 1
            self.message = "You win!"
            print("Player wins!")
            self.save_game_data()
            return
        
        # Check draw
        if self.board_logic.is_draw():
            self.game_over = True
            self.message = "Draw!"
            print("Draw!")
            self.save_game_data()
            return
        
        # AI turn
        self.turn = -1
        self.start_ai_search()
    
    def start_ai_search(self):
        """Start AI search in background thread"""
        if self.ai_running:
            return
        
        print(f"Starting AI search (depth={self.ai_depth}, time={self.ai_time}s)")
        self.ai_running = True
        self.ai_result = None
        
        def search_worker():
            try:
                print(f"[WORKER] Search started")
                move, value, stats = get_best_move(
                    board=self.board_logic,
                    max_depth=self.ai_depth,
                    max_time=self.ai_time,
                    player=-1,
                    use_model=self.use_model,
                    model_path=self.model_path if self.use_model else None,
                    verbose=True
                )
                print(f"[WORKER] Search done: move={move}, value={value:.2f}")
                self.ai_result = (move, value, stats)
            except Exception as e:
                print(f"[WORKER] Error: {e}")
                import traceback
                traceback.print_exc()
                self.ai_result = ("ERROR", None, None)
        
        self.ai_thread = threading.Thread(target=search_worker, daemon=False)
        self.ai_thread.start()
    
    def process_ai_result(self):
        """Check if AI search is done and apply result"""
        if not self.ai_running or self.ai_thread is None:
            return
        
        if self.ai_thread.is_alive():
            # Still searching
            return
        
        # Search is done
        print(f"[MAIN] AI search thread finished")
        self.ai_running = False
        
        if self.ai_result is None:
            print(f"[MAIN] ERROR: No result")
            self.message = "AI Error - No result"
            self.game_over = True
            self.turn = 1
            return
        
        move, value, stats = self.ai_result
        
        if move == "ERROR" or move is None:
            print(f"[MAIN] ERROR: Invalid move")
            self.message = "AI Error - No valid move"
            self.game_over = True
            self.turn = 1
            return
        
        # Apply move
        try:
            r, c = move
            print(f"[MAIN] Playing AI move: ({r}, {c})")
            self.board_logic.play(r, c, -1)

            self.move_history.append((r, c))
            
            if self.board_logic.is_win_from(r, c):
                self.game_over = True
                self.winner = -1
                self.message = "AI wins!"
                print("AI wins!")
                self.save_game_data()
                return
            
            if self.board_logic.is_draw():
                self.game_over = True
                self.message = "Draw!"
                print("Draw!")
                self.save_game_data()
                return
            
            # Player turn
            self.turn = 1
            self.message = f"Your turn - AI played at ({r}, {c})"
            print("Player turn")
        
        except Exception as e:
            print(f"[MAIN] Error applying move: {e}")
            import traceback
            traceback.print_exc()
            self.message = f"Error: {e}"
            self.game_over = True
            self.turn = 1

    def save_game_data(self):
        """Save full game data to data/run_game/"""
        try:
            import numpy as np
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_path = SAVE_DIR / f"game_{timestamp}.json"

            # Build JSON data
            data = {
                "board": self.board_logic.grid.astype(int).tolist(),
                "moves": self.move_history,
                "winner": self.winner,
                "metadata": {
                    "source": "selfplay_gui",
                    "quality": "human",
                    "ai_depth": self.ai_depth,
                    "ai_time": self.ai_time,
                    "use_model": self.use_model,
                    "timestamp": timestamp,
                },
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[INFO] Saved game data → {file_path}")

        except Exception as e:
            print(f"[ERROR] Failed to save game data: {e}")

    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("\n" + "="*60)
        print("Caro Game Started")
        print("="*60)
        print(f"AI: {'CNN Model' if self.use_model else 'Heuristic'}")
        print(f"Depth: {self.ai_depth}, Time: {self.ai_time}s")
        print("="*60 + "\n")
        
        while running:
            # Draw
            self.draw_board()
            pygame.display.flip()
            
            # Check AI result
            if self.turn == -1:
                self.process_ai_result()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_player_move(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset
                        print("Resetting game...")
                        self.__init__(self.use_model, self.model_path, 
                                     self.ai_depth, self.ai_time)
                    elif event.key == pygame.K_q:  # Quit
                        running = False
            
            clock.tick(30)  # 30 FPS
        
        # Cleanup
        if self.ai_thread and self.ai_thread.is_alive():
            print("Waiting for AI thread...")
            self.ai_thread.join(timeout=1.0)
        
        pygame.quit()
        sys.exit()


def main():
    parser = argparse.ArgumentParser(description="Play Caro against AI")
    parser.add_argument('--use_model', action='store_true', 
                       help='Use trained CNN model')
    parser.add_argument('--model_path', type=str, default='checkpoints/caro_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--depth', type=int, default=3,
                       help='AI search depth (1-6)')
    parser.add_argument('--time', type=float, default=2.0,
                       help='AI time limit (seconds)')
    
    args = parser.parse_args()
    
    game = CaroGame(
        use_model=args.use_model,
        model_path=args.model_path,
        ai_depth=args.depth,
        ai_time=args.time
    )
    
    game.run()


if __name__ == "__main__":
    main()