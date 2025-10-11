# sinh dữ liệu bằng AI tự chơi

"""
Sinh dữ liệu tự chơi (self-play) giữa 2 AI (hoặc AI vs random).
Lưu mỗi ván vào data/selfplay/game_xxx.json

Yêu cầu: 
- model.py đã có evaluate_model, policy_suggest
- searchs.py đã có Search (Negamax + AlphaBeta + TT)
"""

import random
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from game import Board
from evaluate import evaluate_pattern
from model import load_model_into_cache, evaluate_model, policy_suggest
from searchs import get_best_move


# Cấu hình cơ bản
SAVE_DIR = "data/selfplay"
os.makedirs(SAVE_DIR, exist_ok=True)


# Chơi 1 ván giữa 2 AI
# ---------------------------------------------------------------------
def play_one_game(game_id=0, max_depth=3, max_time=2.0):
    """
    AI tự chơi với chính nó → sinh 1 file game_{id}.json
    """
    board = Board()
    moves = []
    player = 1
    result = 0
    move_limit = 15 * 15  # giới hạn để tránh loop vô tận

    search = Search(
        board,
        evaluate_fn=evaluate_model,
        policy_fn=policy_suggest,
        max_time=max_time,
    )

    for _ in range(move_limit):
        mv, _ = search.get_best_move(max_depth=max_depth, player=player)
        if mv is None:
            result = 0
            break

        board.play(*mv, player)
        moves.append(list(mv))

        if board.is_win_from(*mv):
            result = player
            break
        if board.is_full():
            result = 0
            break

        player = -player

    game_data = {"moves": moves, "result": int(result)}
    save_path = os.path.join(SAVE_DIR, f"game_{game_id:04d}.json")

    with open(save_path, "w") as f:
        json.dump(game_data, f)

    return game_data


# ---------------------------------------------------------------------
# Sinh nhiều ván tự chơi
# ---------------------------------------------------------------------
def generate_selfplay_games(n_games=100, max_depth=3, max_time=2.0):
    """
    Sinh nhiều ván AI vs AI và lưu file JSON.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    for i in range(n_games):
        play_one_game(game_id=i, max_depth=max_depth, max_time=max_time)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main(n_games=10, max_depth=3):
    load_model_into_cache("checkpoints/caro_epoch80.pt", use_fp16=True)
    generate_selfplay_games(n_games=n_games, max_depth=max_depth)



main(n_games=10, max_depth=3)