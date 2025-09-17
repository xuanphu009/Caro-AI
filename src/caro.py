import pygame
import sys

# Cấu hình bàn cờ
BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 20
WIDTH = HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN

WHITE, BLACK, GRAY = (255,255,255), (0,0,0), (200,200,200)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Caro Console Style")
board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]  # 0 empty, 1 X, -1 O
turn = 1

def draw_board():
    screen.fill(WHITE)
    # Vẽ lưới
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, GRAY, (MARGIN, MARGIN+i*CELL_SIZE), (WIDTH-MARGIN, MARGIN+i*CELL_SIZE))
        pygame.draw.line(screen, GRAY, (MARGIN+i*CELL_SIZE, MARGIN), (MARGIN+i*CELL_SIZE, HEIGHT-MARGIN))
    # Vẽ quân cờ
    font = pygame.font.SysFont("consolas", 24, bold=True)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != 0:
                text = font.render("X" if board[r][c]==1 else "O", True, BLACK)
                screen.blit(text, (MARGIN+c*CELL_SIZE+10, MARGIN+r*CELL_SIZE+5))

running = True
while running:
    draw_board()
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            col = (x - MARGIN) // CELL_SIZE
            row = (y - MARGIN) // CELL_SIZE
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == 0:
                board[row][col] = turn
                turn *= -1

pygame.quit()
sys.exit()
