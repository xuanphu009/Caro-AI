import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from game import Board
from searchs import get_best_move

def test_search_finds_winning_move():
    b = Board(size=10)
    for i in range(4):
        b.play(0, i, 1)
    # AI phải đánh (0,4) để thắng
    move, _ = get_best_move(b, max_depth=2, player=1)
    assert move == (0, 4)
