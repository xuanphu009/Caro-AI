import pytest
from src.game import Board

def test_win_horizontal():
    b = Board(size=10)
    for i in range(5):
        b.play(0, i, 1)
    assert b.is_win((0,4))

def test_generate_candidates_empty():
    b = Board(size=10)
    moves = b.generate_candidates()
    assert (5,5) in moves  # trung tâm board
