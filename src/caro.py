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

# import math
# import random

# # ====== Cấu hình ======
# BOARD_SIZE = 10   # kích thước bàn cờ (10x10 cho nhẹ, có thể đổi 15)
# WIN_LENGTH = 5    # số quân liên tiếp để thắng
# MAX_DEPTH = 3     # độ sâu tìm kiếm minimax

# # Người chơi: X = máy (MAX), O = người (MIN)
# EMPTY = "."
# PLAYER = "O"
# AI = "X"

# # ====== Biểu diễn bàn cờ ======
# def create_board():
#     return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

# def print_board(board):
#     print("   " + " ".join([str(i).rjust(2) for i in range(BOARD_SIZE)]))
#     for i, row in enumerate(board):
#         print(str(i).rjust(2), " ".join(row))

# # ====== Kiểm tra thắng/thua ======
# def check_win(board, symbol):
#     # kiểm tra hàng, cột, chéo
#     directions = [(1,0), (0,1), (1,1), (1,-1)]
#     for i in range(BOARD_SIZE):
#         for j in range(BOARD_SIZE):
#             if board[i][j] == symbol:
#                 for dx, dy in directions:
#                     count = 1
#                     x, y = i+dx, j+dy
#                     while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == symbol:
#                         count += 1
#                         if count == WIN_LENGTH:
#                             return True
#                         x += dx
#                         y += dy
#     return False

# def check_full(board):
#     return all(cell != EMPTY for row in board for cell in row)

# # ====== Hàm đánh giá trạng thái ======
# def evaluate(board):
#     if check_win(board, AI):
#         return 10000
#     if check_win(board, PLAYER):
#         return -10000
    
#     score = 0
#     score += count_sequences(board, AI, 4) * 100
#     score += count_sequences(board, AI, 3) * 50
#     score += count_sequences(board, AI, 2) * 10

#     score -= count_sequences(board, PLAYER, 4) * 100
#     score -= count_sequences(board, PLAYER, 3) * 50
#     score -= count_sequences(board, PLAYER, 2) * 10
#     return score

# def count_sequences(board, symbol, length):
#     """Đếm số chuỗi liên tiếp dài 'length' của symbol"""
#     count = 0
#     directions = [(1,0), (0,1), (1,1), (1,-1)]
#     for i in range(BOARD_SIZE):
#         for j in range(BOARD_SIZE):
#             if board[i][j] == symbol:
#                 for dx, dy in directions:
#                     seq_len = 1
#                     x, y = i+dx, j+dy
#                     while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == symbol:
#                         seq_len += 1
#                         x += dx
#                         y += dy
#                     if seq_len == length:
#                         count += 1
#     return count

# # ====== Minimax với Alpha-Beta ======
# def minimax(board, depth, alpha, beta, maximizingPlayer):
#     if depth == 0 or check_win(board, AI) or check_win(board, PLAYER) or check_full(board):
#         return evaluate(board)

#     if maximizingPlayer:  # MAX = AI
#         maxEval = -math.inf
#         for move in generate_moves(board):
#             i, j = move
#             board[i][j] = AI
#             eval = minimax(board, depth-1, alpha, beta, False)
#             board[i][j] = EMPTY
#             maxEval = max(maxEval, eval)
#             alpha = max(alpha, eval)
#             if beta <= alpha:
#                 break
#         return maxEval
#     else:  # MIN = người
#         minEval = math.inf
#         for move in generate_moves(board):
#             i, j = move
#             board[i][j] = PLAYER
#             eval = minimax(board, depth-1, alpha, beta, True)
#             board[i][j] = EMPTY
#             minEval = min(minEval, eval)
#             beta = min(beta, eval)
#             if beta <= alpha:
#                 break
#         return minEval

# def best_move(board):
#     bestVal = -math.inf
#     move = None
#     for i, j in generate_moves(board):
#         board[i][j] = AI
#         moveVal = minimax(board, MAX_DEPTH-1, -math.inf, math.inf, False)
#         board[i][j] = EMPTY
#         if moveVal > bestVal:
#             bestVal = moveVal
#             move = (i, j)
#     return move

# def generate_moves(board):
#     """Sinh các nước đi hợp lệ (ở gần quân cờ hiện tại để giảm số lượng)"""
#     moves = []
#     for i in range(BOARD_SIZE):
#         for j in range(BOARD_SIZE):
#             if board[i][j] == EMPTY:
#                 # lọc bớt: chỉ chọn ô gần quân đã có (tối ưu)
#                 if has_neighbor(board, i, j):
#                     moves.append((i, j))
#     if not moves:  # nếu bàn rỗng thì đi ngẫu nhiên
#         moves = [(BOARD_SIZE//2, BOARD_SIZE//2)]
#     return moves

# def has_neighbor(board, i, j, distance=1):
#     """Kiểm tra ô (i,j) có quân cờ nào xung quanh không"""
#     for dx in range(-distance, distance+1):
#         for dy in range(-distance, distance+1):
#             x, y = i+dx, j+dy
#             if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
#                 if board[x][y] != EMPTY:
#                     return True
#     return False

# # ====== Vòng lặp chơi ======
# def play():
#     board = create_board()
#     print_board(board)

#     while True:
#         # Người chơi đi
#         x, y = map(int, input("Nhập nước đi của bạn (row col): ").split())
#         if board[x][y] != EMPTY:
#             print("Ô này đã có rồi, đi lại!")
#             continue
#         board[x][y] = PLAYER
#         print_board(board)
#         if check_win(board, PLAYER):
#             print("Bạn thắng!")
#             break
#         if check_full(board):
#             print("Hòa!")
#             break

#         # Máy đi
#         print("Máy đang suy nghĩ...")
#         move = best_move(board)
#         if move:
#             board[move[0]][move[1]] = AI
#         print_board(board)
#         if check_win(board, AI):
#             print("Máy thắng!")
#             break
#         if check_full(board):
#             print("Hòa!")
#             break

# if __name__ == "__main__":
#     play()
