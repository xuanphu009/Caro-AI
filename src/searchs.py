# thuật toán Negamax + Alpha-Beta + TT

from math import inf

def evaluate_simple(board, player):
    """Lượng giá đơn giản: số quân của mình - đối thủ"""
    my_count = (board.grid == player).sum()
    opp_count = (board.grid == -player).sum()
    return my_count - opp_count

def negamax(board, depth, alpha, beta, player, evaluate_fn=evaluate_simple):
    if depth == 0 or board.is_win(board.last_move) or board.is_draw():
        return evaluate_fn(board, player)

    best = -inf
    for r, c in board.generate_candidates():
        board.play(r, c, player)
        val = -negamax(board, depth-1, -beta, -alpha, -player, evaluate_fn)
        board.undo()
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best

def get_best_move(board, depth, player, evaluate_fn=evaluate_simple):
    best_val = -inf
    best_move = None
    for r, c in board.generate_candidates():
        board.play(r, c, player)
        val = -negamax(board, depth-1, -inf, inf, -player, evaluate_fn)
        board.undo()
        if val > best_val:
            best_val = val
            best_move = (r, c)
    return best_move
