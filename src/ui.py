# # giao diện

# ui.py
import pygame

# UI CONFIG (match board size 15x15 as in project)
BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 20
WIDTH = HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN

WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (200,200,200)
BLUE = (30,144,255)
RED = (220,20,60)
YELLOW = (255,215,0)
GREEN = (34,139,34)

pygame.init()
FONT = pygame.font.SysFont("consolas", 24, bold=True)
SMALL_FONT = pygame.font.SysFont("consolas", 18)

def draw_board(screen, board, last_move=None, win_line=None):
    """
    board: numpy array or 2D list with values {0,1,-1}
    last_move: (r,c) tuple to highlight last play
    win_line: list of (r,c) to highlight winning five (optional)
    """
    screen.fill(WHITE)

    # grid
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, GRAY, (MARGIN, MARGIN + i*CELL_SIZE), (WIDTH - MARGIN, MARGIN + i*CELL_SIZE))
        pygame.draw.line(screen, GRAY, (MARGIN + i*CELL_SIZE, MARGIN), (MARGIN + i*CELL_SIZE, HEIGHT - MARGIN))

    # stones
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            val = int(board[r][c])
            if val == 1:
                surf = FONT.render("X", True, BLUE)
                screen.blit(surf, (MARGIN + c*CELL_SIZE + 10, MARGIN + r*CELL_SIZE + 5))
            elif val == -1:
                surf = FONT.render("O", True, RED)
                screen.blit(surf, (MARGIN + c*CELL_SIZE + 10, MARGIN + r*CELL_SIZE + 5))

    # last move
    if last_move:
        r, c = last_move
        pygame.draw.rect(screen, YELLOW, (MARGIN + c*CELL_SIZE + 3, MARGIN + r*CELL_SIZE + 3, CELL_SIZE-6, CELL_SIZE-6), 3)

    # win line
    if win_line:
        for (r, c) in win_line:
            pygame.draw.rect(screen, GREEN, (MARGIN + c*CELL_SIZE + 6, MARGIN + r*CELL_SIZE + 6, CELL_SIZE-12, CELL_SIZE-12), 4)

def draw_info(screen, turn, winner):
    """Turn and winner info at bottom"""
    if winner == 0:
        msg = f"Lượt: {'Người (X)' if turn==1 else 'Máy (O)'}"
        color = BLACK
    else:
        msg = f"{'Người' if winner==1 else 'Máy'} thắng!"
        color = GREEN
    surf = SMALL_FONT.render(msg, True, color)
    screen.blit(surf, (MARGIN, HEIGHT - 30))

def draw_help(screen):
    msg = SMALL_FONT.render("Nhấn R: chơi lại | ESC: thoát", True, BLACK)
    screen.blit(msg, (MARGIN, 6))



# import pygame

# # THIẾT LẬP CƠ BẢN

# BOARD_SIZE = 15
# CELL_SIZE = 40
# MARGIN = 20
# WIDTH = HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN

# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# GRAY = (200, 200, 200)
# RED = (220, 20, 60)
# BLUE = (30, 144, 255)
# GREEN = (34, 139, 34)
# YELLOW = (255, 215, 0)

# pygame.init()
# FONT = pygame.font.SysFont("consolas", 24, bold=True)
# SMALL_FONT = pygame.font.SysFont("consolas", 18)


# # HÀM VẼ

# def draw_board(screen, board, last_move=None, winner=None):
#     """Vẽ bàn cờ và quân cờ."""
#     screen.fill(WHITE)

#     # Lưới caro
#     for i in range(BOARD_SIZE):
#         pygame.draw.line(screen, GRAY, (MARGIN, MARGIN+i*CELL_SIZE), (WIDTH-MARGIN, MARGIN+i*CELL_SIZE))
#         pygame.draw.line(screen, GRAY, (MARGIN+i*CELL_SIZE, MARGIN), (MARGIN+i*CELL_SIZE, HEIGHT-MARGIN))

#     # Quân cờ
#     for r in range(BOARD_SIZE):
#         for c in range(BOARD_SIZE):
#             if board[r][c] == 1:  # Người chơi X
#                 text = FONT.render("X", True, BLUE)
#                 screen.blit(text, (MARGIN + c*CELL_SIZE + 10, MARGIN + r*CELL_SIZE + 5))
#             elif board[r][c] == -1:  # Máy O
#                 text = FONT.render("O", True, RED)
#                 screen.blit(text, (MARGIN + c*CELL_SIZE + 10, MARGIN + r*CELL_SIZE + 5))

#     # Đánh dấu nước đi cuối
#     if last_move:
#         r, c = last_move
#         pygame.draw.rect(screen, YELLOW,
#                          (MARGIN + c*CELL_SIZE + 2, MARGIN + r*CELL_SIZE + 2, CELL_SIZE-4, CELL_SIZE-4), 3)

#     # Nếu có người thắng, highlight toàn bộ
#     if winner != 0:
#         pygame.draw.rect(screen, GREEN, (5, 5, WIDTH-10, HEIGHT-10), 5)
#     # Highlight đường thắng
#     # if win_line:
#     #     for (r, c) in win_line:
#     #         pygame.draw.rect(screen, GREEN, 
#     #                          (MARGIN + c*CELL_SIZE + 5, MARGIN + r*CELL_SIZE + 5, CELL_SIZE-10, CELL_SIZE-10), 4)

# def draw_info(screen, turn, winner):
#     """Hiển thị lượt chơi hoặc kết quả."""
#     if winner == 0:
#         msg = f"Lượt hiện tại: {'Người (X)' if turn==1 else 'Máy (O)'}"
#         color = BLACK
#     else:
#         msg = f"{'Người' if winner==1 else 'Máy'} thắng!"
#         color = GREEN
#     text = SMALL_FONT.render(msg, True, color)
#     screen.blit(text, (MARGIN, HEIGHT - 30))

# def draw_buttons(screen):
#     """Vẽ nút hướng dẫn đơn giản."""
#     msg = SMALL_FONT.render("Nhấn [R] để chơi lại, [ESC] để thoát", True, BLACK)
#     screen.blit(msg, (MARGIN, 5))


# def draw_board(screen, board, last_move=None, win_line=None):
#     """Vẽ bàn cờ, quân cờ, nước đi cuối, và highlight thắng nếu có."""
#     screen.fill(WHITE)
    
#     # Lưới
#     for i in range(BOARD_SIZE):
#         pygame.draw.line(screen, GRAY, (MARGIN, MARGIN + i*CELL_SIZE), (WIDTH-MARGIN, MARGIN + i*CELL_SIZE))
#         pygame.draw.line(screen, GRAY, (MARGIN + i*CELL_SIZE, MARGIN), (MARGIN + i*CELL_SIZE, HEIGHT-MARGIN))

#     # Quân cờ
#     for r in range(BOARD_SIZE):
#         for c in range(BOARD_SIZE):
#             if board[r][c] == 1:
#                 text = FONT.render("X", True, BLUE)
#                 screen.blit(text, (MARGIN + c*CELL_SIZE + 10, MARGIN + r*CELL_SIZE + 5))
#             elif board[r][c] == -1:
#                 text = FONT.render("O", True, RED)
#                 screen.blit(text, (MARGIN + c*CELL_SIZE + 10, MARGIN + r*CELL_SIZE + 5))

#     # Highlight nước đi cuối
#     if last_move:
#         r, c = last_move
#         pygame.draw.rect(screen, YELLOW, 
#                          (MARGIN + c*CELL_SIZE + 2, MARGIN + r*CELL_SIZE + 2, CELL_SIZE-4, CELL_SIZE-4), 3)

#     # Highlight đường thắng
#     if win_line:
#         for (r, c) in win_line:
#             pygame.draw.rect(screen, GREEN, 
#                              (MARGIN + c*CELL_SIZE + 5, MARGIN + r*CELL_SIZE + 5, CELL_SIZE-10, CELL_SIZE-10), 4)
