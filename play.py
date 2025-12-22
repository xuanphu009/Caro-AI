import os
import sys
import json
import time
from datetime import datetime
from gui.interface import *
from source.search import *
from gui.button import Button
import source.utils as utils
import source.gomoku as gomoku
import pygame
from source.search import GomokuAI, CNN_GomokuAI
# Game initializer function
# Link interface with gomoku moves and AI
# Script to be run

# Global game history tracker
class GameHistory:
    def __init__(self):
        self.moves = []
        self.start_time = None
        self.ai_mode = None
        self.first_player = None
        
    def start_game(self, ai_mode, first_player):
        """Start tracking a new game"""
        self.moves = []
        self.start_time = time.time()
        self.ai_mode = ai_mode
        self.first_player = first_player
        
    def add_move(self, row, col, player):
        """Add a move to history"""
        self.moves.append({
            'position': [row, col],
            'player': player,
            'move_number': len(self.moves) + 1
        })
    
    def save_game(self, winner, save_dir="data/run_game"):
        """Save game to JSON file"""
        if not self.moves:
            return None
            
        # Create directory if not exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Count existing games to generate ID
        existing_files = [f for f in os.listdir(save_dir) if f.startswith('game_') and f.endswith('.json')]
        game_id = len(existing_files)
        
        # Calculate game duration
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Prepare game data
        game_data = {
            'game_id': game_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ai_mode': self.ai_mode,
            'first_player': self.first_player,
            'moves': [[m['position'][0], m['position'][1]] for m in self.moves],
            'total_moves': len(self.moves),
            'winner': winner,
            'duration_seconds': round(duration, 2),
            'metadata': {
                'board_size': 15,
                'move_details': self.moves
            }
        }
        
        # Save to file
        filename = f"game_{game_id:04d}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Game saved: {filepath}")
        print(f"   - Moves: {len(self.moves)}")
        print(f"   - Winner: {self._get_winner_name(winner)}")
        print(f"   - Duration: {duration:.1f}s")
        
        return filepath
    
    def _get_winner_name(self, winner):
        """Convert winner code to readable name"""
        if winner == 1:
            return "AI"
        elif winner == -1:
            return "Human"
        else:
            return "Draw"

# Global instance
game_history = GameHistory()


pygame.init()

def selectAIMode():
    """Display AI mode selection menu and return selected AI"""
    pygame.init()
    screen = pygame.display.set_mode((540, 540))
    pygame.display.set_caption('Play Gomoku!')
    
    # Load assets
    board = pygame.image.load(os.path.join("assets", 'board.jpg')).convert()
    menu_board_img = pygame.image.load(os.path.join("assets", "menu_board.png")).convert_alpha()
    button_surf = pygame.image.load(os.path.join("assets", "button.png")).convert_alpha()
    button_surf = pygame.transform.scale(button_surf, (150, 65))
    
    screen.blit(board, (0, 0))
    
    # Draw menu
    menu_board = pygame.transform.scale(menu_board_img, (450, 200))
    menu_board_rect = menu_board.get_rect(center=screen.get_rect().center)
    
    # Title
    title_font = pygame.font.SysFont("arial", 26, bold=True)
    title_text = title_font.render('SELECT AI MODE', True, 'white')
    title_size = title_text.get_size()
    menu_board.blit(title_text, (225 - title_size[0]//2, 20))
    
    # Subtitle
    subtitle_font = pygame.font.SysFont("arial", 16)
    subtitle_text = subtitle_font.render('Choose how AI will play', True, (200, 200, 200))
    subtitle_size = subtitle_text.get_size()
    menu_board.blit(subtitle_text, (225 - subtitle_size[0]//2, 55))
    
    screen.blit(menu_board, menu_board_rect)
    
    # Create buttons
    button_negamax = Button(button_surf, 170, 280, "NEGAMAX", 20)
    button_cnn = Button(button_surf, 360, 280, "CNN", 20)
    
    button_negamax.draw(screen)
    button_cnn.draw(screen)
    
    pygame.display.update()
    
    # Wait for selection
    selected_mode = None
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                
                if button_negamax.rect.collidepoint(mouse_pos):
                    selected_mode = "negamax"
                    run = False
                    print("✓ Selected: Negamax AI (Depth=4)")
                    
                elif button_cnn.rect.collidepoint(mouse_pos):
                    selected_mode = "cnn"
                    run = False
                    print("✓ Selected: CNN Hybrid AI")
    
    # Create AI based on selection
    if selected_mode == "cnn":
        try:
            ai = CNN_GomokuAI(model_path="checkpoints/caro_best.pt")
            ai.ai_mode = "CNN"
        except Exception as e:
            print(f"⚠️  Failed to load CNN: {e}")
            print("⚠️  Falling back to Negamax AI")
            ai = GomokuAI(depth=4)
            ai.ai_mode = "Negamax"
    else:
        ai = GomokuAI(depth=4)
        ai.ai_mode = "Negamax"
    
    return ai

def startGame():
    pygame.init()
    
    # Select AI mode through GUI
    ai = selectAIMode()
    game = GameUI(ai)
    
    # Main game loop - allows multiple games without restarting program
    play_again = True
    while play_again:
        # Reset for new game
        game.ai.resetGame()
        game.resetUI()
        
        button_black = Button(game.buttonSurf, 200, 290, "BLACK", 22)
        button_white = Button(game.buttonSurf, 340, 290, "WHITE", 22)

        # Draw the starting menu
        game.drawMenu()
        game.drawButtons(button_black, button_white, game.screen)
        
        run = True
        color_chosen = False
        
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    play_again = False
                    pygame.quit()
                    return
                    
                if event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0] and not color_chosen:
                    mouse_pos = pygame.mouse.get_pos()
                    # Check which color the user has chosen and set the states
                    game.checkColorChoice(button_black, button_white, mouse_pos)
                    
                    # Only proceed if a color was actually chosen
                    if game.ai.turn != 0:
                        color_chosen = True
                        game.screen.blit(game.board, (0,0))
                        pygame.display.update()
                        
                        # Determine who plays first
                        first_player = "AI" if game.ai.turn == 1 else "Human"
                        
                        # Start tracking game
                        game_history.start_game(
                            ai_mode=getattr(game.ai, 'ai_mode', 'Negamax'),
                            first_player=first_player
                        )
                        
                        if game.ai.turn == 1:
                            game.ai.firstMove()
                            game.drawPiece('black', game.ai.currentI, game.ai.currentJ)
                            pygame.display.update()
                            # Record AI's first move
                            game_history.add_move(game.ai.currentI, game.ai.currentJ, 1)
                            game.ai.turn *= -1
                        
                        # Play the main game
                        main(game)

                        # When the game ends, handle restart
                        if game.ai.checkResult() != None:
                            restart_choice = endMenu(game)
                            if restart_choice:
                                # User chose YES - restart game
                                print("\n" + "="*50)
                                print("Starting new game...")
                                print("="*50 + "\n")
                                run = False  # Exit inner loop to restart
                            else:
                                # User chose NO - exit completely
                                run = False
                                play_again = False
                                pygame.quit()
                                return
                                
            pygame.display.update()   

    pygame.quit()

def endMenu(game):
    """
    Display end game menu and wait for user choice
    Returns True if user wants to play again, False otherwise
    """
    pygame.init()
    last_screen = game.screen.copy()
    game.screen.blit(last_screen, (0,0))
    
    # Check if it's a tie or there's a winner
    result = game.ai.checkResult()
    
    # Save game to file
    game_history.save_game(winner=result)
    
    if result == 0:
        game.drawResult(tie=True)
    else:
        game.drawResult(tie=False)
    
    # Draw buttons
    yes_button = Button(game.buttonSurf, 200, 155, "YES", 18)
    no_button = Button(game.buttonSurf, 350, 155, "NO", 18)
    game.drawButtons(yes_button, no_button, game.screen)
    
    # Wait for user to click Yes or No
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
                
            if event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                
                if yes_button.rect.collidepoint(mouse_pos):
                    print('✓ Selected YES - Restarting game...')
                    return True
                    
                if no_button.rect.collidepoint(mouse_pos):
                    print('✓ Selected NO - Exiting game...')
                    return False
        
        pygame.display.update()
    
    return False


### Main game play loop ###
def main(game):
    pygame.init()
    end = False
    result = game.ai.checkResult()

    while not end:
        turn = game.ai.turn
        color = game.colorState[turn] # black or white depending on player's choice
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            # AI's turn
            if turn == 1:
                move_i, move_j = gomoku.ai_move(game.ai)
                # Make the move and update zobrist hash
                game.ai.setState(move_i, move_j, turn)
                game.ai.rollingHash ^= game.ai.zobristTable[move_i][move_j][0]
                game.ai.emptyCells -= 1

                game.drawPiece(color, move_i, move_j)
                
                # Record AI move
                game_history.add_move(move_i, move_j, turn)
                
                result = game.ai.checkResult()
                # Switch turn
                game.ai.turn *= -1

            # Human's turn
            if turn == -1:
                if event.type == pygame.MOUSEBUTTONDOWN\
                        and pygame.mouse.get_pressed()[0]:
                    # Get human move played
                    mouse_pos = pygame.mouse.get_pos()
                    human_move = utils.pos_pixel2map(mouse_pos[0], mouse_pos[1])
                    move_i = human_move[0]
                    move_j = human_move[1]
                    # print(mouse_pos, move_i, move_j)

                    # Check the validity of human's move
                    if game.ai.isValid(move_i, move_j):
                        game.ai.boardValue = game.ai.evaluate(move_i, move_j, game.ai.boardValue, -1, game.ai.nextBound)
                        game.ai.updateBound(move_i, move_j, game.ai.nextBound)
                        game.ai.currentI, game.ai.currentJ = move_i, move_j
                        # Make the move and update zobrist hash
                        game.ai.setState(move_i, move_j, turn)
                        game.ai.rollingHash ^= game.ai.zobristTable[move_i][move_j][1]
                        game.ai.emptyCells -= 1
                        
                        game.drawPiece(color, move_i, move_j)
                        
                        # Record human move
                        game_history.add_move(move_i, move_j, turn)
                        
                        result =  game.ai.checkResult()
                        game.ai.turn *= -1
            
            if result != None:
                # End game
                end = True



if __name__ == '__main__':
    startGame()