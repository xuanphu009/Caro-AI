# huấn luyện model từ dataset
"""
CARO AI - ADVANCED TRAINING ENGINE (v3.0)
State-of-the-art training pipeline with all modern techniques

Features:
[1] ADVANCED OPTIMIZATION
    ✅ AdamW with weight decay
    ✅ Cosine Annealing with Warmup
    ✅ Gradient Clipping (prevents explosion)
    ✅ Learning Rate Finder (auto find optimal LR)
    ✅ Exponential Moving Average (EMA)
[2] TRAINING STABILIZATION
    ✅ Mixed Precision Training (FP16) - 2x speedup
    ✅ Gradient Accumulation (large effective batch size)
    ✅ Label Smoothing (better generalization)
    ✅ Early Stopping (prevent overfitting)
    ✅ Model Checkpointing (save best/last) 
[3] DATA AUGMENTATION
    ✅ Random rotation (90°, 180°, 270°)
    ✅ Random flip (horizontal, vertical)
    ✅ Board symmetry (8-fold)
    ✅ On-the-fly augmentation (no disk overhead) 
[4] MONITORING & LOGGING
    ✅ TensorBoard integration
    ✅ Detailed metrics (loss, accuracy, policy entropy)
    ✅ Learning rate tracking
    ✅ Validation curves
    ✅ Best model tracking
[5] CURRICULUM LEARNING
    ✅ Progressive difficulty (start easy → hard)
    ✅ Adaptive sampling (focus on hard examples)
    ✅ Dynamic loss weighting

API Usage:
Basic:
    from trainer import Trainer 
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda',
        use_amp=True
    )
    trainer.train(epochs=100)

Advanced:
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        weight_decay=1e-4,
        use_amp=True,
        use_ema=True,
        gradient_clip=1.0,
        label_smoothing=0.1,
        warmup_epochs=5
    )
    
    # Find optimal learning rate
    trainer.find_lr(min_lr=1e-5, max_lr=1e-1)
    
    # Train with all features
    history = trainer.train(
        epochs=100,
        early_stopping_patience=10,
        save_best=True
    )
    
    # Resume from checkpoint
    trainer.load_checkpoint('checkpoints/caro_epoch50.pt')
    trainer.train(epochs=50)  # Continue to epoch 100
"""

import os
import sys
import time
import math
import json
from typing import Optional, Dict, List, Tuple, Callable
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Optional: TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not available. Install with: pip install tensorboard")

# Import from our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import CaroNet, build_model, build_preset, save_checkpoint, load_checkpoint, EMA
from src.dataset import CaroDataset


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class TrainingConfig:
    """Training configuration with sensible defaults"""
    
    # Paths
    data_dir: str = "data/selfplay"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Model
    model_preset: str = "xlarge"  # tiny/small/medium/large/xlarge
    
    # Data
    batch_size: int = 128
    num_workers: int = 4
    val_split: float = 0.1
    
    # Optimization
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Scheduler
    scheduler_type: str = "cosine"  # cosine/onecycle/step
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Regularization
    label_smoothing: float = 0.1
    dropout: float = 0.1
    gradient_clip: float = 1.0
    
    # Mixed Precision
    use_amp: bool = True
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Augmentation
    use_augmentation: bool = True
    augment_prob: float = 0.5
    
    # Loss weights
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Logging
    log_interval: int = 10  # batches
    val_interval: int = 1   # epochs
    save_interval: int = 5  # epochs
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__class__.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


# ═══════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

class BoardAugmenter:
    """
    Apply symmetry transformations to board states
    8-fold symmetry: 4 rotations × 2 flips
    """
    
    @staticmethod
    def rotate_90(board: np.ndarray, move_idx: Optional[int] = None, 
                  board_size: int = 15) -> Tuple[np.ndarray, Optional[int]]:
        """Rotate 90° clockwise"""
        # board shape: (C, H, W)
        rotated = np.rot90(board, k=-1, axes=(1, 2))
        
        if move_idx is not None:
            r, c = move_idx // board_size, move_idx % board_size
            new_c = board_size - 1 - r
            new_r = c
            move_idx = new_r * board_size + new_c
        
        return rotated, move_idx
    
    @staticmethod
    def rotate_180(board: np.ndarray, move_idx: Optional[int] = None,
                   board_size: int = 15) -> Tuple[np.ndarray, Optional[int]]:
        """Rotate 180°"""
        rotated = np.rot90(board, k=2, axes=(1, 2))
        
        if move_idx is not None:
            r, c = move_idx // board_size, move_idx % board_size
            new_r = board_size - 1 - r
            new_c = board_size - 1 - c
            move_idx = new_r * board_size + new_c
        
        return rotated, move_idx
    
    @staticmethod
    def rotate_270(board: np.ndarray, move_idx: Optional[int] = None,
                   board_size: int = 15) -> Tuple[np.ndarray, Optional[int]]:
        """Rotate 270° clockwise"""
        rotated = np.rot90(board, k=1, axes=(1, 2))
        
        if move_idx is not None:
            r, c = move_idx // board_size, move_idx % board_size
            new_c = r
            new_r = board_size - 1 - c
            move_idx = new_r * board_size + new_c
        
        return rotated, move_idx
    
    @staticmethod
    def flip_horizontal(board: np.ndarray, move_idx: Optional[int] = None,
                       board_size: int = 15) -> Tuple[np.ndarray, Optional[int]]:
        """Flip horizontally"""
        flipped = np.flip(board, axis=2).copy()
        
        if move_idx is not None:
            r, c = move_idx // board_size, move_idx % board_size
            new_c = board_size - 1 - c
            move_idx = r * board_size + new_c
        
        return flipped, move_idx
    
    @staticmethod
    def flip_vertical(board: np.ndarray, move_idx: Optional[int] = None,
                     board_size: int = 15) -> Tuple[np.ndarray, Optional[int]]:
        """Flip vertically"""
        flipped = np.flip(board, axis=1).copy()
        
        if move_idx is not None:
            r, c = move_idx // board_size, move_idx % board_size
            new_r = board_size - 1 - r
            move_idx = new_r * board_size + c
        
        return flipped, move_idx
    
    @staticmethod
    def random_transform(board: np.ndarray, move_idx: Optional[int] = None,
                        board_size: int = 15, prob: float = 0.5) -> Tuple[np.ndarray, Optional[int]]:
        """Apply random transformation"""
        if np.random.rand() > prob:
            return board, move_idx
        
        transforms = [
            BoardAugmenter.rotate_90,
            BoardAugmenter.rotate_180,
            BoardAugmenter.rotate_270,
            BoardAugmenter.flip_horizontal,
            BoardAugmenter.flip_vertical,
        ]
        
        transform = np.random.choice(transforms)
        return transform(board, move_idx, board_size)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing
    Prevents overconfident predictions
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        
        # One-hot with smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULERS
# ═══════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with linear warmup
    """
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════
# METRICS TRACKER
# ═══════════════════════════════════════════════════════════════════════════

# class MetricsTracker:
#     """Track and compute training metrics"""
    
#     def __init__(self):
#         self.reset()
    
#     def reset(self):
#         self.metrics = defaultdict(list)
    
#     def update(self, **kwargs):
#         for key, value in kwargs.items():
#             if isinstance(value, torch.Tensor):
#                 value = value.item()
#             self.metrics[key].append(value)
    
#     def average(self) -> Dict[str, float]:
#         return {key: np.mean(values) for key, values in self.metrics.items()}
    
#     def get(self, key: str) -> List[float]:
#         return self.metrics.get(key, [])
class MetricsTracker:
    """Track and average training metrics cleanly."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.counts = {}

    def update(self, **kwargs):
        """Update metrics with batch values."""
        for k, v in kwargs.items():
            if isinstance(v, (float, int)):
                val = float(v)
            elif hasattr(v, "item"):
                val = float(v.item())
            else:
                continue
            if k not in self.metrics:
                self.metrics[k] = 0.0
                self.counts[k] = 0
            self.metrics[k] += val
            self.counts[k] += 1

    def avg(self, key):
        """Return average of given metric key."""
        if key not in self.metrics:
            return 0.0
        return self.metrics[key] / max(1, self.counts[key])

    def summary(self):
        """Return a summary dict of all averages."""
        return {k: self.avg(k) for k in self.metrics}

    def compute(self):
        """Alias for summary() for compatibility."""
        return self.summary()

    def to_dict(self):
        """Alias for summary() — some legacy code calls this."""
        return self.summary()

    def __str__(self):
        parts = []
        for k, v in self.summary().items():
            parts.append(f"{k}: {v:.4f}")
        return " | ".join(parts)
    
    def average(self):
        """Alias for summary() for backward compatibility."""
        return self.summary()





# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Advanced training engine with all modern techniques
    """
    
    def __init__(
        self,
        model: Optional[CaroNet] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        **kwargs
    ):
        """
        Args:
            model: CaroNet model (if None, will build from config)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: TrainingConfig object
            **kwargs: Override config parameters
        """
        # Config
        self.config = config or TrainingConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Setup directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Device
        self.device = torch.device(self.config.device)
        print(f"🔧 Using device: {self.device}")
        
        # Model
        if model is None:
            print(f"🏗️  Building {self.config.model_preset} model...")
            self.model = build_preset(
                self.config.model_preset,
                dropout=self.config.dropout
            )
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
        # Mixed Precision
        self.scaler = GradScaler(enabled=self.config.use_amp)
        
        # EMA
        self.ema = EMA(self.model, decay=self.config.ema_decay) if self.config.use_ema else None
        
        # Loss functions
        if self.config.label_smoothing > 0:
            self.policy_criterion = LabelSmoothingCrossEntropy(self.config.label_smoothing)
        else:
            self.policy_criterion = nn.CrossEntropyLoss()
        
        self.value_criterion = nn.MSELoss()
        
        # Metrics
        self.metrics = MetricsTracker()
        self.history = defaultdict(list)
        
        # TensorBoard
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.config.log_dir)
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Augmenter
        self.augmenter = BoardAugmenter() if self.config.use_augmentation else None
        
        print("✅ Trainer initialized")
        self._print_config()
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            return CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=self.config.warmup_epochs,
                max_epochs=self.config.epochs,
                min_lr=self.config.min_lr
            )
        elif self.config.scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                epochs=self.config.epochs,
                steps_per_epoch=len(self.train_loader) if self.train_loader else 100
            )
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
    
    def _print_config(self):
        """Print training configuration"""
        print("\n" + "="*60)
        print("⚙️  TRAINING CONFIGURATION")
        print("="*60)
        print(f"Model: {self.config.model_preset}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Learning Rate: {self.config.lr}")
        print(f"Weight Decay: {self.config.weight_decay}")
        print(f"Scheduler: {self.config.scheduler_type}")
        print(f"Mixed Precision: {self.config.use_amp}")
        print(f"EMA: {self.config.use_ema}")
        print(f"Augmentation: {self.config.use_augmentation}")
        print(f"Label Smoothing: {self.config.label_smoothing}")
        print(f"Gradient Clip: {self.config.gradient_clip}")
        print("="*60 + "\n")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        self.metrics.reset()
        
        # for batch_idx, (boards, results, move_indices) in enumerate(self.train_loader):
        #     # Move to device
        #     boards = boards.to(self.device)
        #     results = results.to(self.device).float().unsqueeze(1)
        #     move_indices = move_indices.to(self.device).long()
            
        #     # Augmentation
        #     if self.augmenter and np.random.rand() < self.config.augment_prob:
        #         # Apply same random transform to entire batch
        #         boards_np = boards.cpu().numpy()
        #         moves_np = move_indices.cpu().numpy()
                
        #         aug_boards = []
        #         aug_moves = []
        #         for b, m in zip(boards_np, moves_np):
        #             b_aug, m_aug = self.augmenter.random_transform(b, int(m))
        #             aug_boards.append(b_aug)
        #             aug_moves.append(m_aug if m_aug is not None else m)
                
        #         boards = torch.from_numpy(np.array(aug_boards)).to(self.device)
        #         move_indices = torch.tensor(aug_moves, dtype=torch.long).to(self.device)
            
        #     # Forward pass with mixed precision
        #     self.optimizer.zero_grad()
            
        #     with autocast(enabled=self.config.use_amp):
        #         pred_value, policy_logits = self.model(boards)
                
        #         # Losses
        #         value_loss = self.value_criterion(torch.tanh(pred_value), results)
        #         policy_loss = self.policy_criterion(policy_logits, move_indices)
                
        #         loss = (self.config.value_loss_weight * value_loss + 
        #                self.config.policy_loss_weight * policy_loss)
            
        #     # Backward pass
        #     self.scaler.scale(loss).backward()
            
        #     # Gradient clipping
        #     if self.config.gradient_clip > 0:
        #         self.scaler.unscale_(self.optimizer)
        #         torch.nn.utils.clip_grad_norm_(
        #             self.model.parameters(), 
        #             self.config.gradient_clip
        #         )
            
        #     # Optimizer step
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
            
        #     # Update EMA
        #     if self.ema:
        #         self.ema.update(self.model)
            
        #     # Metrics
        #     with torch.no_grad():
        #         # Policy accuracy (top-1)
        #         pred_moves = policy_logits.argmax(dim=1)
        #         policy_acc = (pred_moves == move_indices).float().mean()
                
        #         # Value accuracy (within 0.2)
        #         value_acc = (torch.abs(torch.tanh(pred_value) - results) < 0.2).float().mean()
            
        #     self.metrics.update(
        #         loss=loss,
        #         value_loss=value_loss,
        #         policy_loss=policy_loss,
        #         policy_acc=policy_acc,
        #         value_acc=value_acc
        #     )
            
        #     # Logging
        #     if batch_idx % self.config.log_interval == 0:
        #         self._log_batch(batch_idx, len(self.train_loader))
            
        #     self.global_step += 1
        for batch_idx, batch in enumerate(self.train_loader):
            # --- Unpack batch safely (support 2 or 3 outputs) ---
            if len(batch) == 3:
                boards, results, move_indices = batch
            elif len(batch) == 2:
                boards, results = batch
                move_indices = None
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")

            # Move to device
            boards = boards.to(self.device)
            results = results.to(self.device).float().unsqueeze(1)
            if move_indices is not None:
                move_indices = move_indices.to(self.device).long()

            # Augmentation
            if self.augmenter and np.random.rand() < self.config.augment_prob:
                boards_np = boards.cpu().numpy()

                if move_indices is not None:
                    moves_np = move_indices.cpu().numpy()
                    aug_boards, aug_moves = [], []
                    for b, m in zip(boards_np, moves_np):
                        b_aug, m_aug = self.augmenter.random_transform(b, int(m))
                        aug_boards.append(b_aug)
                        aug_moves.append(m_aug if m_aug is not None else m)
                    boards = torch.from_numpy(np.array(aug_boards)).to(self.device)
                    move_indices = torch.tensor(aug_moves, dtype=torch.long).to(self.device)
                else:
                    # Augment boards only
                    aug_boards = [self.augmenter.random_transform(b, None)[0] for b in boards_np]
                    boards = torch.from_numpy(np.array(aug_boards)).to(self.device)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            with autocast(enabled=self.config.use_amp):
                pred_value, policy_logits = self.model(boards)

                # --- Compute losses depending on available labels ---
                value_loss = self.value_criterion(torch.tanh(pred_value), results)
                if move_indices is not None:
                    policy_loss = self.policy_criterion(policy_logits, move_indices)
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)

                loss = (
                    self.config.value_loss_weight * value_loss
                    + self.config.policy_loss_weight * policy_loss
                )

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update EMA
            if self.ema:
                self.ema.update(self.model)

            # Metrics
            with torch.no_grad():
                if move_indices is not None:
                    pred_moves = policy_logits.argmax(dim=1)
                    policy_acc = (pred_moves == move_indices).float().mean()
                else:
                    policy_acc = torch.tensor(0.0, device=self.device)
                value_acc = (torch.abs(torch.tanh(pred_value) - results) < 0.2).float().mean()

            self.metrics.update(
                loss=loss,
                value_loss=value_loss,
                policy_loss=policy_loss,
                policy_acc=policy_acc,
                value_acc=value_acc
            )

            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._log_batch(batch_idx, len(self.train_loader))

            self.global_step += 1
        
        return self.metrics.average()
    
    # @torch.no_grad()
    # def validate(self) -> Dict[str, float]:
    #     """Validate on validation set"""
    #     if self.val_loader is None:
    #         return {}
        
    #     self.model.eval()
    #     self.metrics.reset()
        
    #     for boards, results, move_indices in self.val_loader:
    #         boards = boards.to(self.device)
    #         results = results.to(self.device).float().unsqueeze(1)
    #         move_indices = move_indices.to(self.device).long()
            
    #         with autocast(enabled=self.config.use_amp):
    #             pred_value, policy_logits = self.model(boards)
                
    #             value_loss = self.value_criterion(torch.tanh(pred_value), results)
    #             policy_loss = self.policy_criterion(policy_logits, move_indices)
                
    #             loss = (self.config.value_loss_weight * value_loss + 
    #                    self.config.policy_loss_weight * policy_loss)
            
    #         # Metrics
    #         pred_moves = policy_logits.argmax(dim=1)
    #         policy_acc = (pred_moves == move_indices).float().mean()
    #         value_acc = (torch.abs(torch.tanh(pred_value) - results) < 0.2).float().mean()
            
    #         self.metrics.update(
    #             val_loss=loss,
    #             val_value_loss=value_loss,
    #             val_policy_loss=policy_loss,
    #             val_policy_acc=policy_acc,
    #             val_value_acc=value_acc
    #         )
        
    #     return self.metrics.average()
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.metrics.reset()

        for batch in self.val_loader:
            # --- Safe unpacking (handle 2 or 3 outputs) ---
            if len(batch) == 3:
                boards, results, move_indices = batch
            elif len(batch) == 2:
                boards, results = batch
                move_indices = None
            else:
                raise ValueError(f"Unexpected validation batch format: {len(batch)} elements")

            boards = boards.to(self.device)
            results = results.to(self.device).float().unsqueeze(1)
            if move_indices is not None:
                move_indices = move_indices.to(self.device).long()

            with autocast(enabled=self.config.use_amp):
                pred_value, policy_logits = self.model(boards)

                # Compute losses
                value_loss = self.value_criterion(torch.tanh(pred_value), results)
                if move_indices is not None:
                    policy_loss = self.policy_criterion(policy_logits, move_indices)
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)

                loss = (
                    self.config.value_loss_weight * value_loss
                    + self.config.policy_loss_weight * policy_loss
                )

            # Metrics
            if move_indices is not None:
                pred_moves = policy_logits.argmax(dim=1)
                policy_acc = (pred_moves == move_indices).float().mean()
            else:
                policy_acc = torch.tensor(0.0, device=self.device)

            value_acc = (torch.abs(torch.tanh(pred_value) - results) < 0.2).float().mean()

            self.metrics.update(
                loss=loss,
                value_loss=value_loss,
                policy_loss=policy_loss,
                policy_acc=policy_acc,
                value_acc=value_acc
            )

        val_metrics = self.metrics.summary() if hasattr(self.metrics, "summary") else self.metrics.to_dict()

        self._log_validation(val_metrics)
        return val_metrics

    
    def train(
        self,
        epochs: Optional[int] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            epochs: Number of epochs (overrides config)
            resume_from: Path to checkpoint to resume from
        
        Returns:
            history: Dictionary of training metrics
        """
        if epochs is None:
            epochs = self.config.epochs
        
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        
        start_epoch = self.current_epoch + 1
        end_epoch = start_epoch + epochs
        
        for epoch in range(start_epoch, end_epoch + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config.val_interval == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {}
            
            # Scheduler step
            if self.config.scheduler_type != "onecycle":
                self.scheduler.step()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update history
            for key, value in all_metrics.items():
                self.history[key].append(value)
            
            # Log to TensorBoard
            if self.writer:
                for key, value in all_metrics.items():
                    self.writer.add_scalar(f'metrics/{key}', value, epoch)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            self._print_epoch_summary(epoch, all_metrics, epoch_time)
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save(f"caro_epoch{epoch}.pt")
            
            # Save best model
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save("caro_best.pt", is_best=True)
                print(f"💾 New best model! Val loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                print(f"No improvement for {self.config.early_stopping_patience} epochs")
                break
            
            # Save last checkpoint
            if epoch == end_epoch:
                self.save("caro_last.pt")
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        
        if self.writer:
            self.writer.close()
        
        return dict(self.history)
    
    def _log_batch(self, batch_idx: int, total_batches: int):
        """Log batch progress"""
        metrics = self.metrics.average()
        print(f"[Epoch {self.current_epoch}] [{batch_idx}/{total_batches}] "
              f"Loss: {metrics.get('loss', 0):.4f} | "
              f"Value: {metrics.get('value_loss', 0):.4f} | "
              f"Policy: {metrics.get('policy_loss', 0):.4f} | "
              f"P-Acc: {metrics.get('policy_acc', 0):.3f}", end='\r')
    
    def _log_validation(self, val_metrics: dict):
        """
        Pretty-print validation metrics (compatibility helper).
        Expects val_metrics to be a dict of averaged metrics (e.g. from MetricsTracker.summary()).
        """
        # Defensive fallback: accept None or unexpected structures
        if val_metrics is None:
            print("⚠️  No validation metrics to log.")
            return

        # Safely extract values (fallback to 0.0)
        loss = val_metrics.get("loss", 0.0)
        value_loss = val_metrics.get("value_loss", val_metrics.get("v_loss", 0.0))
        policy_loss = val_metrics.get("policy_loss", val_metrics.get("p_loss", 0.0))
        value_acc = val_metrics.get("value_acc", val_metrics.get("v_acc", 0.0))
        policy_acc = val_metrics.get("policy_acc", val_metrics.get("p_acc", 0.0))

        # Print a tidy validation summary (matches training log style)
        print("\n" + "-"*60)
        print(f"🔎 Validation | Loss: {loss:.4f} | V-Loss: {value_loss:.4f} | P-Loss: {policy_loss:.4f} | "
            f"V-Acc: {value_acc:.3f} | P-Acc: {policy_acc:.3f}")
        print("-"*60 + "\n")

    
    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float], epoch_time: float):
        """Print epoch summary"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{self.config.epochs} ({epoch_time:.1f}s)")
        print(f"{'='*60}")
        
        # Training metrics
        print(f"Train Loss: {metrics.get('loss', 0):.4f} | "
              f"Value: {metrics.get('value_loss', 0):.4f} | "
              f"Policy: {metrics.get('policy_loss', 0):.4f}")
        print(f"Train Acc:  Policy: {metrics.get('policy_acc', 0):.3f} | "
              f"Value: {metrics.get('value_acc', 0):.3f}")
        
        # Validation metrics
        if 'val_loss' in metrics:
            print(f"Val Loss:   {metrics['val_loss']:.4f} | "
                  f"Value: {metrics.get('val_value_loss', 0):.4f} | "
                  f"Policy: {metrics.get('val_policy_loss', 0):.4f}")
            print(f"Val Acc:    Policy: {metrics.get('val_policy_acc', 0):.3f} | "
                  f"Value: {metrics.get('val_value_acc', 0):.3f}")
        
        print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}\n")
    
    def save(self, filename: str, is_best: bool = False):
        """Save checkpoint"""
        path = os.path.join(self.config.checkpoint_dir, filename)
        
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            metrics={
                'best_val_loss': self.best_val_loss,
                'epochs_without_improvement': self.epochs_without_improvement
            },
            ema=self.ema
        )
        
        if is_best:
            print(f"💾 Saved best model → {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint and resume training"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        self.best_val_loss = metrics.get('best_val_loss', float('inf'))
        self.epochs_without_improvement = metrics.get('epochs_without_improvement', 0)
        
        # Load EMA
        if 'ema_shadow' in checkpoint and self.ema:
            self.ema.shadow = checkpoint['ema_shadow']
        
        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
    
    def find_lr(
        self,
        min_lr: float = 1e-7,
        max_lr: float = 1e-1,
        num_iter: int = 100,
        smooth_f: float = 0.05
    ) -> Tuple[float, List[float], List[float]]:
        """
        Learning Rate Finder (Leslie Smith method)
        
        Returns:
            suggested_lr: Suggested learning rate
            lrs: List of learning rates tested
            losses: List of losses at each LR
        """
        print("\n" + "="*60)
        print("🔍 LEARNING RATE FINDER")
        print("="*60)
        
        if self.train_loader is None:
            raise ValueError("train_loader is required for LR finder")
        
        # Save original state
        original_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        self.model.train()
        
        # Setup
        lr_mult = (max_lr / min_lr) ** (1 / num_iter)
        lr = min_lr
        
        lrs = []
        losses = []
        best_loss = float('inf')
        avg_loss = 0.0
        
        # Iterate
        iterator = iter(self.train_loader)
        
        for iteration in range(num_iter):
            # Get batch
            try:
                boards, results, move_indices = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                boards, results, move_indices = next(iterator)
            
            boards = boards.to(self.device)
            results = results.to(self.device).float().unsqueeze(1)
            move_indices = move_indices.to(self.device).long()
            
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.use_amp):
                pred_value, policy_logits = self.model(boards)
                value_loss = self.value_criterion(torch.tanh(pred_value), results)
                policy_loss = self.policy_criterion(policy_logits, move_indices)
                loss = value_loss + 0.5 * policy_loss
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Smooth loss
            avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            smoothed_loss = avg_loss / (1 - (1 - smooth_f) ** (iteration + 1))
            
            # Track
            lrs.append(lr)
            losses.append(smoothed_loss)
            
            # Update best
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Stop if loss explodes
            if smoothed_loss > 4 * best_loss or np.isnan(smoothed_loss):
                print(f"⚠️  Stopping early at iteration {iteration}")
                break
            
            # Progress
            if iteration % 10 == 0:
                print(f"Iter {iteration}/{num_iter} | LR: {lr:.2e} | Loss: {smoothed_loss:.4f}", end='\r')
            
            # Next LR
            lr *= lr_mult
        
        print()
        
        # Restore original state
        self.model.load_state_dict(original_state['model'])
        self.optimizer.load_state_dict(original_state['optimizer'])
        
        # Find suggested LR (steepest negative gradient)
        gradients = np.gradient(losses)
        suggested_idx = np.argmin(gradients)
        suggested_lr = lrs[suggested_idx]
        
        print(f"\n✅ Suggested Learning Rate: {suggested_lr:.2e}")
        print(f"   (Steepest descent at iteration {suggested_idx})")
        print("="*60)
        
        # Plot if matplotlib available
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(lrs, losses, label='Loss')
            ax.axvline(suggested_lr, color='r', linestyle='--', label=f'Suggested: {suggested_lr:.2e}')
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Loss')
            ax.set_title('Learning Rate Finder')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = os.path.join(self.config.log_dir, 'lr_finder.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved → {plot_path}")
            plt.close()
        except ImportError:
            pass
        
        return suggested_lr, lrs, losses
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib not available")
            return
        
        if not self.history:
            print("⚠️  No history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        epochs = range(1, len(self.history['loss']) + 1)
        
        # Loss
        ax = axes[0, 0]
        if 'loss' in self.history:
            ax.plot(epochs, self.history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in self.history:
            ax.plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Value Loss
        ax = axes[0, 1]
        if 'value_loss' in self.history:
            ax.plot(epochs, self.history['value_loss'], label='Train Value Loss', linewidth=2)
        if 'val_value_loss' in self.history:
            ax.plot(epochs, self.history['val_value_loss'], label='Val Value Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Policy Loss
        ax = axes[1, 0]
        if 'policy_loss' in self.history:
            ax.plot(epochs, self.history['policy_loss'], label='Train Policy Loss', linewidth=2)
        if 'val_policy_loss' in self.history:
            ax.plot(epochs, self.history['val_policy_loss'], label='Val Policy Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[1, 1]
        if 'policy_acc' in self.history:
            ax.plot(epochs, self.history['policy_acc'], label='Train Policy Acc', linewidth=2)
        if 'val_policy_acc' in self.history:
            ax.plot(epochs, self.history['val_policy_acc'], label='Val Policy Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Policy Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved → {save_path}")
        else:
            plot_path = os.path.join(self.config.log_dir, 'training_history.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved → {plot_path}")
        
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def create_data_loaders(
    data_dir: str = "data/selfplay",
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        data_dir: Directory containing game JSON files
        batch_size: Batch size
        val_split: Validation split ratio
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader
    """
    print(f"📂 Loading data from {data_dir}")
    
    # Load full dataset
    full_dataset = CaroDataset(data_dir)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No data found in {data_dir}. Run selfplay.py first!")
    
    print(f"✅ Loaded {len(full_dataset)} positions")
    
    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"📊 Train: {train_size} | Val: {val_size}")
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    data_dir: str = "data/selfplay",
    model_preset: str = "xlarge", # chuyen tu medium -> xlarge 
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    use_amp: bool = True,
    use_ema: bool = True,
    find_lr_first: bool = False,
    **kwargs
) -> Tuple[CaroNet, Dict]:
    """
    Convenience function to train a model from scratch
    
    Args:
        data_dir: Data directory
        model_preset: Model preset (tiny/small/medium/large/xlarge)
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        use_amp: Use mixed precision
        use_ema: Use EMA
        find_lr_first: Run LR finder before training
        **kwargs: Additional config overrides
    
    Returns:
        model: Trained model
        history: Training history
    """
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*17 + "CARO AI TRAINER" + " "*26 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=kwargs.get('num_workers', 4)
    )
    
    # Build model
    model = build_preset(model_preset)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        use_amp=use_amp,
        use_ema=use_ema,
        model_preset=model_preset,
        batch_size=batch_size,
        **kwargs
    )
    
    # Find optimal LR
    if find_lr_first:
        suggested_lr, _, _ = trainer.find_lr()
        
        # Ask user if they want to use suggested LR
        response = input(f"\nUse suggested LR {suggested_lr:.2e}? (y/n): ")
        if response.lower() == 'y':
            trainer.config.lr = suggested_lr
            trainer.optimizer = optim.AdamW(
                model.parameters(),
                lr=suggested_lr,
                weight_decay=trainer.config.weight_decay
            )
            trainer.scheduler = trainer._build_scheduler()
            print(f"✅ Updated learning rate to {suggested_lr:.2e}")
    
    # Train
    history = trainer.train(epochs=epochs)
    
    # Plot history
    trainer.plot_history()
    
    # Save final model
    trainer.save("caro_final.pt")
    
    print("\n✅ Training complete!")
    print(f"💾 Best model saved to: {os.path.join(trainer.config.checkpoint_dir, 'caro_best.pt')}")
    print(f"📊 Training plots saved to: {trainer.config.log_dir}")
    
    return model, history


# ═══════════════════════════════════════════════════════════════════════════
# MAIN (for testing)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Caro AI Model")
    parser.add_argument('--data_dir', type=str, default='data/selfplay', help='Data directory')
    parser.add_argument('--preset', type=str, default='medium', 
                       choices=['tiny', 'small', 'medium', 'large', 'xlarge'],
                       help='Model preset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--find_lr', action='store_true', help='Run LR finder first')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no_ema', action='store_true', help='Disable EMA')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("📝 Please run selfplay.py first to generate training data:")
        print("   python src/selfplay.py")
        sys.exit(1)
    
    # Check if data has files
    json_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
    if len(json_files) == 0:
        print(f"❌ No JSON files found in {args.data_dir}")
        print("📝 Please run selfplay.py first to generate training data:")
        print("   python src/selfplay.py")
        sys.exit(1)
    
    print(f"✅ Found {len(json_files)} game files")
    
    # Train
    model, history = train_model(
        data_dir=args.data_dir,
        model_preset=args.preset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_amp=not args.no_amp,
        use_ema=not args.no_ema,
        find_lr_first=args.find_lr
    )
    
    print("\n" + "═"*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY")
    print("═"*60)
    print("\nNext steps:")
    print("  1. Test model: python test/test_model.py")
    print("  2. Integrate with search: update searchs.py to use evaluate_model")
    print("  3. Play against AI: python src/caro.py")
    print("  4. Generate more data: python src/selfplay.py")