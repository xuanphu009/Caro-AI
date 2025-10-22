from turtle import mode
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


pygame.init()

def startGame():
    pygame.init()
    # Initializations
    mode = input("Chọn chế độ AI (1=Minimax, 2=CNN): ")

    if mode.strip() == "2":
        ai = CNN_GomokuAI(model_path="checkpoints/caro_best.pt")
    else:
        ai = GomokuAI(depth=4)
        
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
                        
                        if game.ai.turn == 1:
                            game.ai.firstMove()
                            game.drawPiece('black', game.ai.currentI, game.ai.currentJ)
                            pygame.display.update()
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
                        result =  game.ai.checkResult()
                        game.ai.turn *= -1
            
            if result != None:
                # End game
                end = True



if __name__ == '__main__':
    startGame()