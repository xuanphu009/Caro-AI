# định nghĩa CNN (PyTorch)

"""
CARO AI - TRUNG TÂM MÔ HÌNH TIÊN TIẾN (v3.0)
Kiến trúc CNN tiên tiến để đánh giá vị trí Gomoku/Caro

[1] TRÍCH XUẤT FEATURES ĐA TỶ LỆ
    - Convolutions song song (1x1, 3x3, 5x5) theo Inception
    - Bắt được cả local patterns và global strategy

[2] RESIDUAL TOWER VỚI SE-NET ATTENTION
    - Deep residual blocks (10-20 lớp)
    - Squeeze-Excitation để tập trung vào channels quan trọng
    - Bottleneck design tiết kiệm tham số

[3] OUTPUT NHỊ KỲ (Dual-Head)
    - Value Head: Đánh giá giá trị vị trí [-1, 1]
    - Policy Head: Phân phối xác suất nước đi [H*W]

[4] HUẤN LUYỆN TIÊN TIẾN
    - Mixed Precision (FP16) tăng tốc độ 2x
    - Label Smoothing cải thiện tổng quát hóa
    - Cosine Annealing + Warmup điều chỉnh learning rate
    - Gradient Clipping & Weight Decay tránh overfitting
    - EMA (Exponential Moving Average) cho inference ổn định

[5] INFERENCE TỐI ƯU
    - TorchScript/ONNX export cho deploy
    - Batch evaluation cho search tree nodes
    - FP16 inference trên CUDA
    - Model caching & warmup

API Overview:
Building (Xây dựng):
  model = build_model(arch='resnet', base_channels=128, n_blocks=12)
  model = build_preset('large')  # tiny, small, medium, large, xlarge

Training (Huấn luyện):
  from trainer import Trainer
  trainer = Trainer(model, train_loader, val_loader)
  trainer.train(epochs=100)

Inference (Đánh giá):
  load_model_into_cache(path, use_fp16=True, use_ema=True)
  score = evaluate_model(board, player=1)  # [-1, 1]
  moves = policy_suggest(board, top_k=20)  # {(r,c): score}
  values = batch_evaluate(boards, players)  # Nhanh hơn
"""

import os
import sys
import math
import time
from typing import Optional, Dict, Tuple, List, Union
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# GLOBAL CONFIG
BOARD_SIZE = 15 # Kích thước bàn cờ 15x15
DEFAULT_CHECKPOINT = "checkpoints/caro_best.pt" # Đường dẫn checkpoint mặc định
# Chọn GPU nếu có, nếu không dùng CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BUILDING BLOCKS
class SEBlock(nn.Module):
    """
    Khối Squeeze-and-Excitation (SENet)
    
    Cơ chế hoạt động:
    1. SQUEEZE: Nén toàn bộ kênh thành một giá trị (lấy trung bình)
    2. EXCITATION: Học tỷ trọng quan trọng của mỗi kênh
    3. Re-weight: Nhân lại features với các trọng số này
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    Khối Residual với Bottleneck Design và SE-Attention
    
    Cấu trúc (Bottleneck):
        Input (C channels)
           ↓
        Conv1x1: C → C/4 (nén xuống)
           ↓
        Conv3x3: C/4 → C/4 (lớn nhất)
           ↓
        Conv1x1: C/4 → C (mở rộng lại)
           ↓
        SE Block (re-weight channels)
           ↓
        Add input + output (residual connection)
           ↓
        ReLU
           ↓
        Output
    
    Lợi ích:
    - Giảm tham số từ Conv3x3 thông thường
    - Cho phép huấn luyện mạng sâu
    - SE Block giúp tập trung vào thông tin quan trọng
    """
    def __init__(self, channels: int, bottleneck_ratio: float = 0.25, use_se: bool = True):
        super().__init__()
        mid_channels = int(channels * bottleneck_ratio)
        
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.se = SEBlock(channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):

        """
        Forward pass với residual connection
        Công thức: output = ReLU(x + bottleneck(x))
        """
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        out += residual
        out = self.relu(out)
        return out


class InceptionModule(nn.Module):
    """
    Khối Inception: 4 đường song song
    - Conv1x1: Capture local patterns
    - Conv3x3: Capture medium-range patterns
    - Conv5x5: Capture large-range patterns
    - MaxPool: Lọc features nổi bật
    - Tất cả kết hợp lại = rich representation
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Distribute channels across paths
        c1 = out_channels // 4
        c3 = out_channels // 4
        c5 = out_channels // 4
        cp = out_channels - c1 - c3 - c5
        
        # 1x1 path
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 path
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 path (factorized as 3x3 + 3x3)
        self.path5 = nn.Sequential(
            nn.Conv2d(in_channels, c5, 3, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5, c5, 3, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True)
        )
        
        # Pool path
        self.pathp = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, cp, 1, bias=False),
            nn.BatchNorm2d(cp),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return torch.cat([
            self.path1(x),
            self.path3(x),
            self.path5(x),
            self.pathp(x)
        ], dim=1)

# MAIN ARCHITECTURE
class CaroNet(nn.Module):
    """
    Mạng CNN chính cho Caro/Gomoku
    Kiến trúc tổng thể:
        Board Input (15x15)
           ↓
        STEM (2 Conv layers) → Trích xuất features ban đầu
           ↓
        TOWER (10-20 Residual/Inception blocks) → Deep feature extraction
           ↓
        ┌─────────────────┴──────────────────┐
        ↓                                    ↓
    POLICY HEAD                         VALUE HEAD
    (Dự đoán nước đi)                  (Dự đoán xác suất thắng)
        ↓                                    ↓
    Softmax over moves                  Tanh ∈ [-1, 1]
        ↓                                    ↓
    Policy distribution                Evaluation score
    
    Inputs:
        - arch: Kiên trúc của tower ('resnet' | 'inception' | 'hybrid')
        - in_channels: Số kênh input (2: player + opponent)
        - board_size: Kích thước bàn (15)
        - base_channels: Số kênh cơ sở (64/96/128/192)
        - n_blocks: Số lớp residual/inception (6-20)
        - use_se: Có dùng SE-blocks không
        - dropout: Tỷ lệ dropout (0.0-0.3)
    
    Outputs:
        - value: (B, 1) ∈ [-1, 1] - xác suất thắng
        - policy: (B, H*W) - logits của từng nước đi
    """
    
    def __init__(
        self,
        arch: str = 'resnet',
        in_channels: int = 2,
        board_size: int = BOARD_SIZE,
        base_channels: int = 128,
        n_blocks: int = 12,
        use_se: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.arch = arch
        self.board_size = board_size
        self.base_channels = base_channels
        
        # [1] STEM: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # [2] TOWER: Deep feature extraction
        if arch == 'resnet':
            self.tower = self._build_resnet_tower(base_channels, n_blocks, use_se)
        elif arch == 'inception':
            self.tower = self._build_inception_tower(base_channels, n_blocks)
        elif arch == 'hybrid':
            self.tower = self._build_hybrid_tower(base_channels, n_blocks, use_se)
        else:
            raise ValueError(f"Unknown arch: {arch}")
        
        # [3] POLICY HEAD: Move distribution prediction
        self.policy_conv = nn.Sequential(
            nn.Conv2d(base_channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * board_size * board_size, board_size * board_size)
        )
        
        # [4] VALUE HEAD: Win probability estimation
        self.value_conv = nn.Sequential(
            nn.Conv2d(base_channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.value_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * board_size * board_size, base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(base_channels, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self._initialize_weights()

    # Hàm xây dựng tower cho tưng kiến trúc
    def _build_resnet_tower(self, channels: int, n_blocks: int, use_se: bool):
        """Build ResNet-style tower"""
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(channels, bottleneck_ratio=0.25, use_se=use_se))
        return nn.Sequential(*blocks)
    
    def _build_inception_tower(self, channels: int, n_blocks: int):
        """Build Inception-style tower"""
        blocks = []
        for _ in range(n_blocks):
            blocks.append(InceptionModule(channels, channels))
        return nn.Sequential(*blocks)
    
    def _build_hybrid_tower(self, channels: int, n_blocks: int, use_se: bool):
        """Build hybrid tower (alternating ResNet + Inception)"""
        blocks = []
        for i in range(n_blocks):
            if i % 2 == 0:
                blocks.append(ResidualBlock(channels, use_se=use_se))
            else:
                blocks.append(InceptionModule(channels, channels))
        return nn.Sequential(*blocks)
    
    def _initialize_weights(self):
        """Kaiming initialization for conv, Xavier for linear"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 2, H, W) board state
        Returns:
            value: (B, 1) in [-1, 1]
            policy: (B, H*W) raw logits
        """
        # Feature extraction
        features = self.stem(x)
        features = self.tower(features)
        
        # Policy head
        p = self.policy_conv(features)
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)
        
        # Value head
        v = self.value_conv(features)
        v = v.view(v.size(0), -1)
        value = self.value_fc(v)
        
        return value, policy
    
    def get_config(self) -> dict:
        """Return model configuration for saving/loading"""
        return {
            'arch': self.arch,
            'board_size': self.board_size,
            'base_channels': self.base_channels,
            'n_blocks': len(self.tower)
        }


# MODEL BUILDER
def build_model(
    arch: str = 'hybrid',
    base_channels: int = 192,
    n_blocks: int = 16,
    **kwargs
) -> CaroNet:
    """
    Build model with preset configurations
    
    Presets:
        - Small: base_channels=64, n_blocks=6 (fast, ~1M params)
        - Medium: base_channels=96, n_blocks=10 (balanced, ~3M params)
        - Large: base_channels=128, n_blocks=12 (strong, ~7M params)
        - XLarge: base_channels=192, n_blocks=16 (very strong, ~18M params)
    """
    model = CaroNet(
        arch=arch,
        base_channels=base_channels,
        n_blocks=n_blocks,
        **kwargs
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    return model


# EXPONENTIAL MOVING AVERAGE (EMA)
class EMA:
    """
    Exponential Moving Average (Trung bình động hàm mũ)
    
    Tác dụng:
    - Giữ bản "bóng" của trọng số mô hình
    - Trong quá trình huấn luyện, update trọng số slowly
    - Dùng trọng số EMA cho inference thay vì trọng số mới nhất
    - Cho kết quả ổn định hơn
    
    Công thức update:
        shadow_t = decay * shadow_{t-1} + (1 - decay) * weight_t
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights (for inference)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """Restore original weights (for training)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


# CHECKPOINT MANAGEMENT
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
    ema: Optional[EMA] = None
):
    """Save comprehensive checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config() if hasattr(model, 'get_config') else {},
        'epoch': epoch,
        'metrics': metrics or {}
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow
    
    torch.save(checkpoint, path)
    print(f"💾 Saved checkpoint → {path}")


def load_checkpoint(
    path: str,
    device: Optional[str] = None,
    use_ema: bool = False
) -> CaroNet:
    """Load checkpoint and return model in eval mode"""
    if device is None:
        device = DEVICE
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Build model from config
    config = checkpoint.get('model_config', {})
    model = build_model(**config) if config else build_model()
    
    # Load weights
    if use_ema and 'ema_shadow' in checkpoint:
        print("📊 Loading EMA weights")
        # Create temporary EMA and apply
        temp_ema = EMA(model)
        temp_ema.shadow = checkpoint['ema_shadow']
        temp_ema.apply_shadow(model)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    # Print metrics if available
    if 'metrics' in checkpoint and checkpoint['metrics']:
        print(f"📈 Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


#Class InferenceENgine
class InferenceEngine:
    """
    Engine đánh giá tối ưu với cache và batch processing
    
    Tính năng:
    - Sử dụng Mixed Precision (FP16) trên GPU
    - Warmup GPU trước khi dùng
    - Batch evaluate hiệu quả
    - Xử lý lỗi mạnh mẽ
    """
    def __init__(self, model: CaroNet, device: str = DEVICE, use_fp16: bool = True):
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16 and device == 'cuda'
        
        self.model.to(device)
        self.model.eval()
        
        if self.use_fp16:
            self.model.half()
        
        self._warmup()
    
    def _warmup(self, n_runs: int = 5):
        """Warmup GPU/CPU"""
        dummy = torch.zeros((4, 2, BOARD_SIZE, BOARD_SIZE), 
                           dtype=torch.float16 if self.use_fp16 else torch.float32,
                           device=self.device)
        with torch.no_grad():
            for _ in range(n_runs):
                self.model(dummy)
    
    @torch.no_grad()
    def evaluate(self, board, player: int = 1) -> float:
        """Single board evaluation - FIXED: More robust"""
        try:
            x = self._board_to_tensor(board, player)
            value, _ = self.model(x)
            score = float(value.item())
            # Clamp to [-1, 1]
            return max(-1.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            return 0.0
    
    @torch.no_grad()
    def policy(self, board, player: int = 1, top_k: Optional[int] = None) -> Dict[Tuple[int,int], float]:
        """Get policy distribution"""
        try:
            x = self._board_to_tensor(board, player)
            _, logits = self.model(x)
            logits = logits.squeeze(0).cpu().numpy()
            
            grid = getattr(board, 'grid', None)
            if grid is None:
                return {}
            
            moves = {}
            for idx, score in enumerate(logits):
                r, c = idx // BOARD_SIZE, idx % BOARD_SIZE
                if grid[r, c] == 0:
                    moves[(r, c)] = float(score)
            
            if top_k is not None:
                items = sorted(moves.items(), key=lambda x: x[1], reverse=True)[:top_k]
                return dict(items)
            
            return moves
        except Exception as e:
            print(f"Policy failed: {e}")
            return {}
    
    @torch.no_grad()
    def batch_evaluate(self, boards: List, players: List[int]) -> np.ndarray:
        """Efficient batch evaluation"""
        if len(boards) != len(players):
            raise ValueError("boards and players must have same length")
        
        try:
            tensors = [self._board_to_tensor(b, p) for b, p in zip(boards, players)]
            X = torch.cat(tensors, dim=0)
            
            values, _ = self.model(X)
            return values.squeeze(1).cpu().numpy()
        except Exception as e:
            print(f"Batch evaluation failed: {e}")
            return np.zeros(len(boards))
    
    def _board_to_tensor(self, board, player: int) -> torch.Tensor:
        """Convert board to model input"""
        try:
            if hasattr(board, 'to_cnn_input'):
                arr = board.to_cnn_input(player)
            else:
                grid = np.array(board.grid if hasattr(board, 'grid') else board, dtype=np.int8)
                p_layer = (grid == player).astype(np.float32)
                o_layer = (grid == -player).astype(np.float32)
                arr = np.stack([p_layer, o_layer], axis=0)
            
            tensor = torch.from_numpy(arr).unsqueeze(0)
            dtype = torch.float16 if self.use_fp16 else torch.float32
            return tensor.to(device=self.device, dtype=dtype)
        except Exception as e:
            print(f"Board to tensor conversion failed: {e}")
            raise


# Global engine cache
_GLOBAL_ENGINE: Optional[InferenceEngine] = None


def load_model_into_cache(
    path: str = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    use_fp16: bool = True,
    use_ema: bool = True
) -> InferenceEngine:
    """Load model into global cache for fast repeated inference"""
    global _GLOBAL_ENGINE
    
    print(f"Loading model from {path}...")
    
    try:
        model = load_checkpoint(path, device=device, use_ema=use_ema)
        _GLOBAL_ENGINE = InferenceEngine(model, device=device or DEVICE, use_fp16=use_fp16)
        print(f"Model loaded successfully")
        return _GLOBAL_ENGINE
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise


def evaluate_model(board, current_player: int = 1) -> float:
    """Evaluate using cached model - FIXED: Better error handling"""
    global _GLOBAL_ENGINE
    
    if _GLOBAL_ENGINE is None:
        raise RuntimeError("No model cached. Call load_model_into_cache() first.")
    
    return _GLOBAL_ENGINE.evaluate(board, current_player)


def policy_suggest(board, model: Optional[CaroNet] = None,
                   top_k: Optional[int] = None,
                   device: Optional[str] = None,
                   use_fp16: bool = False) -> Dict[Tuple[int,int], float]:
    """Get policy suggestions"""
    if model is not None:
        engine = InferenceEngine(model, device=device or DEVICE, use_fp16=use_fp16)
        return engine.policy(board, top_k=top_k)
    
    if _GLOBAL_ENGINE is None:
        raise RuntimeError("No model loaded. Call load_model_into_cache() first.")
    
    return _GLOBAL_ENGINE.policy(board, top_k=top_k)


def batch_evaluate(boards: List, players: List[int],
                   model: Optional[CaroNet] = None,
                   device: Optional[str] = None,
                   use_fp16: bool = False) -> np.ndarray:
    """Batch evaluation for efficiency"""
    if model is not None:
        engine = InferenceEngine(model, device=device or DEVICE, use_fp16=use_fp16)
        return engine.batch_evaluate(boards, players)
    
    if _GLOBAL_ENGINE is None:
        raise RuntimeError("No model loaded. Call load_model_into_cache() first.")
    
    return _GLOBAL_ENGINE.batch_evaluate(boards, players)


# EXPORT UTILITIES
def export_to_torchscript(model: CaroNet, path: str, optimize: bool = True):
    """Export to TorchScript for faster loading"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    model.eval()
    sample = torch.zeros((1, 2, BOARD_SIZE, BOARD_SIZE))
    
    if optimize:
        # Trace with optimizations
        traced = torch.jit.trace(model, sample)
        traced = torch.jit.freeze(traced)
    else:
        traced = torch.jit.script(model)
    
    traced.save(path)
    print(f"Exported TorchScript → {path}")


def export_to_onnx(model: CaroNet, path: str, opset: int = 14):
    """Export to ONNX for cross-platform inference"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    model.eval()
    sample = torch.zeros((1, 2, BOARD_SIZE, BOARD_SIZE))
    
    torch.onnx.export(
        model, sample, path,
        opset_version=opset,
        input_names=['board'],
        output_names=['value', 'policy'],
        dynamic_axes={'board': {0: 'batch'}}
    )
    print(f"Exported ONNX → {path}")


# BENCHMARKING
def benchmark_model(model: CaroNet, board_sizes: List[int] = [1, 10, 100, 500]):
    """Benchmark inference speed"""
    print("\n" + "═"*60)
    print("⚡ MODEL BENCHMARK")
    print("═"*60)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Warmup
    dummy = torch.zeros((10, 2, BOARD_SIZE, BOARD_SIZE), device=device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    
    results = {}
    
    for batch_size in board_sizes:
        x = torch.zeros((batch_size, 2, BOARD_SIZE, BOARD_SIZE), device=device)
        
        # Warmup this batch size
        with torch.no_grad():
            for _ in range(3):
                model(x)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(10):
                model(x)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.perf_counter() - start
        
        avg_time = elapsed / 10
        throughput = batch_size / avg_time
        
        results[batch_size] = {
            'avg_time': avg_time,
            'throughput': throughput,
            'latency_per_sample': avg_time / batch_size * 1000  # ms
        }
        
        print(f"Batch {batch_size:4d}: {avg_time*1000:6.2f}ms | "
              f"{throughput:7.1f} samples/s | "
              f"{results[batch_size]['latency_per_sample']:5.2f}ms/sample")
    
    print("═"*60)
    return results


# MODEL ANALYSIS
def analyze_model(model: CaroNet, verbose: bool = True):
    """Analyze model architecture and parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print("\n" + "═"*60)
        print("MODEL ANALYSIS")
        print("═"*60)
        print(f"Architecture: {model.arch.upper()}")
        print(f"Base Channels: {model.base_channels}")
        print(f"Board Size: {model.board_size}x{model.board_size}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Memory (FP32): {total_params * 4 / 1024**2:.2f} MB")
        print(f"Memory (FP16): {total_params * 2 / 1024**2:.2f} MB")
        
        # Layer-wise breakdown
        print("\nLayer Breakdown:")
        print("-"*60)
        
        layer_params = {}
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            layer_params[name] = params
            pct = params / total_params * 100
            print(f"{name:15s}: {params:10,} params ({pct:5.2f}%)")
        
        print("═"*60)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_params': layer_params if verbose else {}
    }


# VISUALIZATION (Optional)
def visualize_attention(model: CaroNet, board, save_path: Optional[str] = None):
    """
    Visualize what the model pays attention to (requires matplotlib)
    Extracts and visualizes intermediate feature maps
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare input
    if hasattr(board, 'to_cnn_input'):
        arr = board.to_cnn_input(1)
    else:
        grid = np.array(board.grid)
        arr = np.stack([(grid == 1).astype(float), (grid == -1).astype(float)])
    
    x = torch.from_numpy(arr).unsqueeze(0).float().to(device)
    
    # Forward pass and capture intermediate features
    with torch.no_grad():
        features_stem = model.stem(x)
        features_tower = model.tower(features_stem)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Attention Visualization', fontsize=16)
    
    # Input
    axes[0, 0].imshow(arr[0], cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 0].set_title('Input: Player 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(arr[1], cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 1].set_title('Input: Player 2')
    axes[0, 1].axis('off')
    
    # Stem features (average across channels)
    stem_avg = features_stem[0].mean(0).cpu().numpy()
    axes[0, 2].imshow(stem_avg, cmap='viridis')
    axes[0, 2].set_title('Stem Features (avg)')
    axes[0, 2].axis('off')
    
    # Tower features at different depths
    tower_avg = features_tower[0].mean(0).cpu().numpy()
    axes[1, 0].imshow(tower_avg, cmap='viridis')
    axes[1, 0].set_title('Tower Features (avg)')
    axes[1, 0].axis('off')
    
    # Max activation
    tower_max = features_tower[0].max(0)[0].cpu().numpy()
    axes[1, 1].imshow(tower_max, cmap='hot')
    axes[1, 1].set_title('Max Activation')
    axes[1, 1].axis('off')
    
    # Attention heatmap (L2 norm across channels)
    attention = (features_tower[0] ** 2).sum(0).sqrt().cpu().numpy()
    im = axes[1, 2].imshow(attention, cmap='YlOrRd')
    axes[1, 2].set_title('Attention Heatmap')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization → {save_path}")
    else:
        plt.show()


# PRESET CONFIGURATIONS
PRESET_CONFIGS = {
    'tiny': {
        'arch': 'resnet',
        'base_channels': 48,
        'n_blocks': 4,
        'use_se': False,
        'dropout': 0.0
    },
    'small': {
        'arch': 'resnet',
        'base_channels': 64,
        'n_blocks': 6,
        'use_se': True,
        'dropout': 0.1
    },
    'medium': {
        'arch': 'resnet',
        'base_channels': 96,
        'n_blocks': 10,
        'use_se': True,
        'dropout': 0.1
    },
    'large': {
        'arch': 'resnet',
        'base_channels': 144,
        'n_blocks': 14,
        'use_se': True,
        'dropout': 0.1
    },
    'xlarge': {
        'arch': 'resnet',
        'base_channels': 192,
        'n_blocks': 18,
        'use_se': True,
        'dropout': 0.15
    },
    'inception_medium': {
        'arch': 'inception',
        'base_channels': 96,
        'n_blocks': 10,
        'dropout': 0.1
    }
}


def build_preset(preset: str, **overrides) -> CaroNet:
    """
    Build model from preset configuration
    
    Available presets:
        - tiny: ~300K params, fast inference
        - small: ~1M params, good for CPU
        - medium: ~3M params, balanced
        - large: ~7M params, strong performance
        - xlarge: ~18M params, maximum strength
        - inception_medium: Inception architecture
    
    Example:
        model = build_preset('large', n_blocks=14)  # Override n_blocks
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset].copy()
    config.update(overrides)
    
    return build_model(**config)
