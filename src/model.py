# định nghĩa CNN (PyTorch)

# Hàm evaluate_model: nếu có model PyTorch, load và trả giá trị; nếu không, fallback về evaluate_pattern
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np

from game import Board
from evaluate import evaluate_pattern

# Minimal PyTorch network + loader + evaluate wrapper
# Replace SimpleCaroNet with your real architecture if you have one.

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# --- Config (chỉnh nếu cần) ---
BOARD_SIZE = 15
IN_CHANNELS = 2         # thường là [player, opponent]; chỉnh nếu bạn dùng khác
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE
CHECKPOINT_PATH = "models/checkpoint.pt"  # chỉnh đường dẫn nếu bạn lưu chỗ khác

# --- Small CNN with value + policy heads ---
class SimpleCaroNet(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, board_size: int = BOARD_SIZE, hidden: int = 64):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True)

        # a few conv layers
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)

        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden)

        # policy head
        self.policy_conv = nn.Conv2d(hidden, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, POLICY_SIZE)

        # value head
        self.value_conv = nn.Conv2d(hidden, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, hidden)
        self.value_fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # policy
        p = self.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # logits for each board cell -> (B, BOARD*BOARD)

        # value
        v = self.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = self.relu(self.value_fc1(v))
        v = self.value_fc2(v)  # (B,1)

        return v, p  # raw value (unbounded), policy logits

# --- model loader + evaluate wrapper ---
_model: Optional[nn.Module] = None
_device = None

def load_model(path: str = CHECKPOINT_PATH, device: str = "cpu") -> Optional[nn.Module]:
    """
    Load checkpoint (or create network skeleton if no checkpoint).
    Returns model or None if torch unavailable.
    """
    global _model, _device
    if not TORCH_AVAILABLE:
        return None
    if _model is not None:
        return _model
    _device = torch.device(device)
    net = SimpleCaroNet(in_channels=IN_CHANNELS, board_size=BOARD_SIZE)
    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=_device)
            # If state is a dict with 'model_state_dict' adjust accordingly
            if isinstance(state, dict) and "model_state_dict" in state:
                net.load_state_dict(state["model_state_dict"])
            else:
                net.load_state_dict(state)
        except Exception as e:
            # can't load checkpoint => keep freshly initialized net
            print(f"[model] warning: could not load checkpoint {path}: {e}")
    net.eval()
    net.to(_device)
    _model = net
    return _model

def evaluate_model(board, current_player: int = 1) -> float:
    """
    Evaluate board and return a scalar in [-1, 1] from POV of current_player.
    - board: object providing `to_cnn_input(current_player)` -> np.ndarray (C,H,W)
    - If torch not available or model not loaded, fallback to quick heuristic.
    """
    # Heuristic fallback if torch unavailable
    def heuristic_eval(b, player):
        # simple heuristic: difference in counts weighted by adjacent patterns
        arr = b.to_cnn_input(player) if hasattr(b, "to_cnn_input") else None
        if arr is None:
            return 0.0
        # arr shape (C,H,W). assume channel0 = player stones, channel1 = opponent
        try:
            player_count = float(arr[0].sum())
            opp_count = float(arr[1].sum())
            # normalize by board area
            diff = (player_count - opp_count) / (BOARD_SIZE * BOARD_SIZE)
            return float(max(-1.0, min(1.0, diff)))
        except Exception:
            return 0.0

    if not TORCH_AVAILABLE:
        return heuristic_eval(board, current_player)

    model = load_model()
    if model is None:
        return heuristic_eval(board, current_player)

    # prepare input
    x = board.to_cnn_input(current_player)  # expects np.ndarray (C,H,W)
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    # expand batch
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,C,H,W)
    xt = xt.to(_device)
    with torch.no_grad():
        v, p = model(xt)
        # apply tanh to squash value to [-1,1]
        val = float(torch.tanh(v.squeeze()).cpu().item())
    return val

# Quick test helper (useful when debugging)
if __name__ == "__main__" and TORCH_AVAILABLE:
    # quick smoke test: random tensor
    net = SimpleCaroNet()
    inp = torch.randn(2, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    v, p = net(inp)
    print("v.shape", v.shape, "p.shape", p.shape)

