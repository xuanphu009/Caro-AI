# định nghĩa CNN (PyTorch)

# model_stub.py
# Hàm evaluate_model: nếu có model PyTorch, load và trả giá trị; nếu không, fallback về evaluate_pattern
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from game import Board
from evaluate import evaluate_pattern

# Example placeholder: bạn sẽ thay bằng model thực tế của bạn
def evaluate_model(board: Board, current_player: int) -> float:
    """
    Trả value in [-1,1] từ POV của current_player.
    Nếu PyTorch có sẵn và model được load, dùng model; ngược lại fallback về evaluate_pattern.
    """
    if not TORCH_AVAILABLE:
        # fallback: scale pattern heuristic về [-1,1]
        h = evaluate_pattern(board, current_player)
        return max(-1.0, min(1.0, h / 100.0))
    else:
        # TODO: load model once (caching) và forward
        # Dưới đây là template; khi bạn có model thực, sửa phần load/model.forward.
        # Example:
        # state = torch.tensor(board.to_cnn_input(current_player), dtype=torch.float32).unsqueeze(0)
        # with torch.no_grad():
        #     v = model(state)  # giả định trả tensor shape (1,)
        # return float(torch.tanh(v).item())
        return 0.0



