import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from game import Board
from searchs import get_best_move

def test_ai_move_on_empty_board():
    b = Board()
    b.play(7,7,1)
    b.play(7,8,-1)
    move, _ = get_best_move(b, max_depth=3, player=1)
    # assert điều kiện hợp lý, ví dụ move là nước hợp lệ
    assert b.in_bounds(*move)



def test_legal_moves_and_generate_candidates():
    board = Board(5)
    # ban đầu có 25 ô trống
    assert len(board.legal_moves()) == 25
    # nếu chưa có quân nào, generate_candidates đánh giữa
    assert board.generate_candidates() == [(2, 2)]
    # sau khi đánh 1 nước
    board.play(2, 2, 1)
    candidates = board.generate_candidates()
    assert (2, 3) in candidates  # phải có ô lân cận


def test_is_win_horizontal():
    board = Board(5)
    for i in range(5):
        board.play(2, i, 1)
    assert board.is_win_from(2, 2)  # thắng ngang


def test_is_win_vertical():
    board = Board(5)
    for i in range(5):
        board.play(i, 3, -1)
    assert board.is_win_from(2, 3)  # thắng dọc


def test_is_win_diagonal():
    board = Board(5)
    for i in range(5):
        board.play(i, i, 1)
    assert board.is_win_from(4, 4)  # thắng chéo
