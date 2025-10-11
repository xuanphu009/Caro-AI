# định nghĩa CNN (PyTorch)

# Hàm evaluate_model: nếu có model PyTorch, load và trả giá trị; nếu không, fallback về evaluate_pattern
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


# Minimal PyTorch network + loader + evaluate wrapper
# Replace SimpleCaroNet with your real architecture if you have one.

# model.py
"""
Simple yet effective CNN for Caro (value + policy heads).
This file is the canonical definition of the network and the
load/save/evaluate utilities used by trainer.py and searchs.py.

API provided:
- class SimpleCaroNet(in_channels=2, board_size=15, base_channels=64, n_blocks=6)
- build_model(...) -> SimpleCaroNet
- load_checkpoint(path, device=None) -> model
- save_checkpoint(path, model, optimizer=None, epoch=None)
- evaluate_model(board, current_player=1, model=None, device=None) -> float
- policy_suggest(board, model=None, top_k=None, device=None) -> dict{(r,c):score}
- batch_evaluate(boards, players, model=None, device=None) -> np.array(values)
"""

# src/model.py
"""
Optimized model center for Caro AI (value + policy heads).
- Architected for a good tradeoff between strength and inference speed.
- Provides utilities for single/batch evaluation, loading/saving, and exporting.
- Designed as "source-of-truth" for trainer.py, searchs.py, selfplay.py.

Key API:
  - SimpleCaroNet(...)                # model class
  - build_model(...) -> SimpleCaroNet
  - save_checkpoint(path, model, optimizer=None, epoch=None)
  - load_checkpoint(path, device=None) -> model (in eval mode)
  - load_model_into_cache(path, device=None, use_fp16=False)
  - evaluate_model(board, current_player=1, model=None, device=None, use_fp16=False)
  - policy_suggest(board, model=None, top_k=None, device=None, use_fp16=False)
  - batch_evaluate(boards, players, model=None, device=None, use_fp16=False)
  - warmup_model(model, device, dtype=torch.float32)
  - export_onnx(model, sample_input, path, opset=11)

Notes:
  - Input encoding: 2 channels: [player_layer, opponent_layer], shape (B,2,15,15)
  - Value output: raw scalar (apply tanh on use); policy: logits (H*W)
"""

from typing import Optional, Dict, Tuple, List
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 15
CHECKPOINT_DEFAULT = "checkpoints/caro_latest.pt"

# ------------------------------------------------------------------
# Lightweight, efficient residual block (bottleneck-like) for speed
# ------------------------------------------------------------------
class FastResBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 1):
        super().__init__()
        mid = int(channels * expansion)
        # Use small convs, no bias (BN follows)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.act(out + x)

# ------------------------------------------------------------------
# Main network: residual tower + policy & value heads
# ------------------------------------------------------------------
class SimpleCaroNet(nn.Module):
    def __init__(self, in_channels: int = 2, board_size: int = BOARD_SIZE,
                 base_channels: int = 64, n_blocks: int = 6, expansion: float = 1.0):
        """
        Arguments:
          - base_channels: controls model capacity. 64 is a good default for CPU; use 128 on GPU.
          - n_blocks: number of residual blocks. 6–10 is a reasonable range.
          - expansion: internal channel expansion inside block; 1.0 keeps same channels.
        """
        super().__init__()
        self.board_size = board_size
        self.base_channels = base_channels

        # initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # residual tower
        blocks = []
        for _ in range(n_blocks):
            blocks.append(FastResBlock(base_channels, expansion=expansion))
        self.tower = nn.Sequential(*blocks)

        # policy head: small conv -> fc to H*W logits
        self.policy_conv = nn.Conv2d(base_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # value head: small conv -> fc -> scalar
        self.value_conv = nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, base_channels // 2)
        self.value_fc2 = nn.Linear(base_channels // 2, 1)

        self._init_weights()

    def _init_weights(self):
        # He initialization for conv; xavier for linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input x: (B, 2, H, W)
        Returns:
          - value: (B, 1) raw (apply tanh when interpreting)
          - policy_logits: (B, H*W)
        """
        out = self.stem(x)
        out = self.tower(out)

        # policy head
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # value head
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_fc2(v)  # raw scalar

        return v, p

# ------------------------------------------------------------------
# Helper: build model with sensible defaults (exposed)
# ------------------------------------------------------------------
def build_model(in_channels: int = 2, board_size: int = BOARD_SIZE,
                base_channels: int = 64, n_blocks: int = 6, expansion: float = 1.0) -> SimpleCaroNet:
    return SimpleCaroNet(in_channels=in_channels, board_size=board_size,
                         base_channels=base_channels, n_blocks=n_blocks, expansion=expansion)

# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------
def save_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, epoch: Optional[int] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    torch.save(payload, path)

def load_checkpoint(path: str, device: Optional[str] = None) -> SimpleCaroNet:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ck = torch.load(path, map_location=device_t)
    # build default model with same config as saved one (we assume checkpoint fits default architecture).
    model = build_model()
    if isinstance(ck, dict) and "model_state_dict" in ck:
        model.load_state_dict(ck["model_state_dict"])
    elif isinstance(ck, dict):
        model.load_state_dict(ck)
    else:
        raise RuntimeError("Unsupported checkpoint format")
    model.to(device_t)
    model.eval()
    return model

# ------------------------------------------------------------------
# Internal cached model for fast repeated inference calls
# ------------------------------------------------------------------
_INTERNAL = {
    "model": None,
    "device": None,
    "use_fp16": False
}

def load_model_into_cache(path: str = CHECKPOINT_DEFAULT, device: Optional[str] = None, use_fp16: bool = False):
    """
    Load a checkpoint and cache it for evaluate_model / policy_suggest calls.
    use_fp16: if True and CUDA available, convert model to half for faster inference & smaller memory.
    """
    global _INTERNAL
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(path, device=device)
    if use_fp16 and device.startswith("cuda"):
        model.half()
    _INTERNAL["model"] = model
    _INTERNAL["device"] = device
    _INTERNAL["use_fp16"] = use_fp16
    return model

def _get_model_and_device(model: Optional[SimpleCaroNet], device: Optional[str], use_fp16: bool):
    if model is None:
        if _INTERNAL["model"] is None:
            raise RuntimeError("No model loaded; call load_model_into_cache(path) or pass model explicitly")
        model = _INTERNAL["model"]
        device = _INTERNAL["device"]
        use_fp16 = _INTERNAL["use_fp16"]
    else:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    return model, device, use_fp16

# ------------------------------------------------------------------
# Board -> tensor helper (consistent with dataset)
# ------------------------------------------------------------------
def board_to_tensor(board, current_player: int = 1) -> torch.Tensor:
    """
    Convert a Board into tensor shape (1,2,H,W) dtype float32 (or float16 if model.half()).
    Board is expected to either:
      - have method .to_cnn_input(current_player) returning np.array (2,H,W)
      - or attribute .grid: numpy array (H,W) with values {0,1,-1}
    """
    if hasattr(board, "to_cnn_input"):
        arr = np.asarray(board.to_cnn_input(current_player), dtype=np.float32)
    else:
        grid = np.array(board.grid, dtype=np.int8)
        player_layer = (grid == current_player).astype(np.float32)
        opp_layer = (grid == -current_player).astype(np.float32)
        arr = np.stack([player_layer, opp_layer], axis=0)
    # shape -> (1,2,H,W)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor

# ------------------------------------------------------------------
# Inference APIs
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(board, current_player: int = 1,
                   model: Optional[SimpleCaroNet] = None,
                   device: Optional[str] = None,
                   use_fp16: bool = False) -> float:
    """
    Return a scalar in [-1,1] from POV of current_player.
    Use cached model if model param is None.
    """
    model, device, use_fp16 = _get_model_and_device(model, device, use_fp16)
    x = board_to_tensor(board, current_player)
    dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else torch.float32
    x = x.to(device=device, dtype=dtype)
    # if model is half, ensure inputs are half
    if use_fp16 and hasattr(model, "half"):
        try:
            model.half()
        except Exception:
            pass
    v_raw, _ = model(x)  # (1,1) and (1,H*W)
    # convert to float and tanh scale
    v = torch.tanh(v_raw).squeeze().cpu().item()
    return float(v)

@torch.no_grad()
def policy_suggest(board, model: Optional[SimpleCaroNet] = None,
                   top_k: Optional[int] = None,
                   device: Optional[str] = None,
                   use_fp16: bool = False) -> Dict[Tuple[int,int], float]:
    """
    Return dict {(r,c): score} for empty cells only.
    If top_k provided, returns only top_k moves.
    Scores are raw logits (higher = better).
    """
    model, device, use_fp16 = _get_model_and_device(model, device, use_fp16)
    x = board_to_tensor(board, current_player=1)
    dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else torch.float32
    x = x.to(device=device, dtype=dtype)
    if use_fp16 and hasattr(model, "half"):
        try:
            model.half()
        except Exception:
            pass
    _, logits = model(x)
    logits = logits.squeeze(0).cpu().numpy()  # (H*W,)
    w = model.board_size
    candidates = {}
    grid = getattr(board, "grid", None)
    for idx, val in enumerate(logits):
        r = idx // w
        c = idx % w
        if grid is None or grid[r, c] == 0:
            candidates[(r, c)] = float(val)
    if top_k is not None:
        items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(items)
    return candidates

@torch.no_grad()
def batch_evaluate(boards: List, players: List[int],
                   model: Optional[SimpleCaroNet] = None,
                   device: Optional[str] = None,
                   use_fp16: bool = False) -> np.ndarray:
    """
    Efficient batch evaluation. Returns np.array (N,) of values in [-1,1].
    boards: list of Board objects
    players: matching list of current_player (1 or -1)
    """
    if len(boards) != len(players):
        raise ValueError("boards and players must match length")
    model, device, use_fp16 = _get_model_and_device(model, device, use_fp16)
    tensors = []
    for b, p in zip(boards, players):
        t = board_to_tensor(b, p)
        tensors.append(t)
    X = torch.cat(tensors, dim=0)
    dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else torch.float32
    X = X.to(device=device, dtype=dtype)
    if use_fp16 and hasattr(model, "half"):
        try:
            model.half()
        except Exception:
            pass
    v_raw, _ = model(X)
    v = torch.tanh(v_raw).squeeze(1).cpu().numpy()
    return v

# ------------------------------------------------------------------
# Utilities: warmup & export
# ------------------------------------------------------------------
def warmup_model(model: SimpleCaroNet, device: Optional[str] = None, dtype=torch.float32, n_runs: int = 3):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    B = 2
    sample = torch.zeros((B, 2, model.board_size, model.board_size), dtype=dtype, device=device)
    with torch.no_grad():
        for _ in range(n_runs):
            model(sample)

def export_onnx(model: SimpleCaroNet, sample_input: torch.Tensor, path: str, opset: int = 11):
    """Export model to ONNX (policy + value outputs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.eval()
    torch.onnx.export(model, sample_input, path, opset_version=opset,
                      input_names=["input"], output_names=["value", "policy"], dynamic_axes={"input": {0: "batch"}})

def to_torchscript(model: SimpleCaroNet, sample_input: torch.Tensor, path: str):
    """Trace and save TorchScript for faster startup on CPU/GPU."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.eval()
    traced = torch.jit.trace(model, sample_input)
    traced.save(path)

# ------------------------------------------------------------------
# Fast test main
# ------------------------------------------------------------------
if __name__ == "__main__":
    # quick smoke test to ensure forward works
    m = build_model(base_channels=64, n_blocks=6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(device)
    x = torch.zeros((2, 2, BOARD_SIZE, BOARD_SIZE), device=device)
    with torch.no_grad():
        v, p = m(x)
    print("smoke: v.shape", v.shape, "p.shape", p.shape)
