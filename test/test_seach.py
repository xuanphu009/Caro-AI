from src.game import Board
from src.searchs import get_best_move

def test_search_finds_winning_move():
    b = Board(size=10)
    for i in range(4):
        b.play(0, i, 1)
    # AI phải đánh (0,4) để thắng
    move = get_best_move(b, depth=2, player=1)
    assert move == (0,4)
