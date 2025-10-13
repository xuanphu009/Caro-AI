# main.py
import pygame, sys
from ui import draw_board, draw_info, draw_help, WIDTH, HEIGHT, CELL_SIZE, MARGIN, BOARD_SIZE
from api import CaroController

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Caro - Người vs Máy (AI mạnh nhất)")

# Tùy chỉnh model path + depth/time
MODEL_PATH = "checkpoints/caro_best.pt"   # đổi theo checkpoint bạn có
SEARCH_DEPTH = 4
SEARCH_TIME = 2.0

controller = CaroController(model_path=MODEL_PATH, max_depth=SEARCH_DEPTH, max_time=SEARCH_TIME)

clock = pygame.time.Clock()
running = True

while running:
    draw_board(screen, controller.board.grid if hasattr(controller.board, "grid") else controller.board, 
               last_move=controller.last_move, win_line=controller.win_line)
    draw_info(screen, controller.turn, controller.winner)
    draw_help(screen)
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                controller.reset()

        elif event.type == pygame.MOUSEBUTTONDOWN and controller.turn == 1 and controller.winner == 0:
            x, y = event.pos
            col = (x - MARGIN) // CELL_SIZE
            row = (y - MARGIN) // CELL_SIZE
            # Ensure integer
            try:
                row = int(row); col = int(col)
            except:
                continue
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                placed = controller.human_move(row, col)
                if placed and controller.winner == 0:
                    # Call AI synchronously (blocks until done)
                    controller.ai_move()

    clock.tick(30)

pygame.quit()
sys.exit()
