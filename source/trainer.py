# trainer_fixed.py - SỬA LỖI & TỐI ƯU
# ✅ Fix MetricsTracker API consistency
# ✅ Tối ưu training loop
# ✅ Thêm class weighting cho imbalanced data
# ✅ Improved validation logic

import os
import sys
import time
import math
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import CaroNet, build_preset, save_checkpoint, EMA
from dataset import CaroDataset


# ═══════════════════════════════════════════════════════════════════════════
# ✅ FIXED: Consistent MetricsTracker
# ═══════════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """
    ✅ FIXED: Unified API with both average() and summary()
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        """Update metrics with batch values"""
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
        """Return average of given metric"""
        if key not in self.metrics:
            return 0.0
        return self.metrics[key] / max(1, self.counts[key])
    
    def summary(self):
        """Return dict of all averages"""
        return {k: self.avg(k) for k in self.metrics}
    
    def average(self):
        """✅ Alias for backward compatibility"""
        return self.summary()
    
    def to_dict(self):
        """✅ Alias for compatibility"""
        return self.summary()
    
    def __str__(self):
        parts = [f"{k}: {v:.4f}" for k, v in self.summary().items()]
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# ✅ IMPROVED: Label Smoothing with better defaults
# ═══════════════════════════════════════════════════════════════════════════

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


# ═══════════════════════════════════════════════════════════════════════════
# ✅ IMPROVED: Weighted MSE for value head (handle imbalanced labels)
# ═══════════════════════════════════════════════════════════════════════════

class WeightedMSELoss(nn.Module):
    """
    ✅ NEW: Weighted MSE to handle label imbalance
    Upweight rare labels (e.g., if more wins than losses)
    """
    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute weights based on sign
        weights = torch.ones_like(target)
        weights[target > 0] = self.pos_weight
        weights[target < 0] = self.neg_weight
        
        # Weighted MSE
        mse = (pred - target) ** 2
        return (weights * mse).mean()


# ═══════════════════════════════════════════════════════════════════════════
# ✅ IMPROVED: Cosine Warmup Scheduler
# ═══════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup"""
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════
# ✅ MAIN TRAINER (FIXED & OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    ✅ FIXED & OPTIMIZED Trainer
    - Fixed MetricsTracker API
    - Added class weighting
    - Improved validation
    - Better early stopping
    """
    
    def __init__(
        self,
        model: Optional[CaroNet] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
        use_ema: bool = True,
        label_smoothing: float = 0.1,
        gradient_clip: float = 1.0,
        warmup_epochs: int = 5,
        epochs: int = 100,
        device: str = None,
        checkpoint_dir: str = "checkpoints",
        **kwargs
    ):
        # Device
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"🔧 Using device: {self.device}")
        
        # Model
        if model is None:
            print(f"🏗️  Building xlarge model...")
            self.model = build_preset("xlarge")
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=epochs,
            min_lr=1e-6
        )
        
        # Mixed Precision
        self.scaler = GradScaler(enabled=use_amp)
        self.use_amp = use_amp
        
        # EMA
        self.ema = EMA(self.model, decay=0.999) if use_ema else None
        
        # ✅ IMPROVED: Compute class weights from training data
        self.value_criterion = self._build_value_criterion(train_loader)
        
        # Policy loss
        if label_smoothing > 0:
            self.policy_criterion = LabelSmoothingCrossEntropy(label_smoothing)
        else:
            self.policy_criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.metrics = MetricsTracker()
        self.history = defaultdict(list)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Config
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print("✅ Trainer initialized")
    
    def _build_value_criterion(self, train_loader):
        """
        ✅ NEW: Compute class weights from training data
        """
        if train_loader is None:
            return nn.MSELoss()
        
        # Sample labels from training set
        labels = []
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) >= 2:
                labels.append(batch[1].numpy())
            if batch_idx >= 10:  # Sample first 10 batches
                break
        
        if len(labels) == 0:
            return nn.MSELoss()
        
        labels = np.concatenate(labels)
        
        # Compute weights
        n_pos = np.sum(labels > 0)
        n_neg = np.sum(labels < 0)
        total = len(labels)
        
        if n_pos == 0 or n_neg == 0:
            print("⚠️  No class imbalance detected, using standard MSE")
            return nn.MSELoss()
        
        # Inverse frequency weighting
        pos_weight = total / (2 * n_pos)
        neg_weight = total / (2 * n_neg)
        
        print(f"📊 Class weights: pos={pos_weight:.2f}, neg={neg_weight:.2f}")
        return WeightedMSELoss(pos_weight=pos_weight, neg_weight=neg_weight)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        self.metrics.reset()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack batch
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
            
            # Forward
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                pred_value, policy_logits = self.model(boards)
                
                # Value loss
                value_loss = self.value_criterion(torch.tanh(pred_value), results)
                
                # Policy loss
                if move_indices is not None:
                    policy_loss = self.policy_criterion(policy_logits, move_indices)
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                loss = value_loss + 0.5 * policy_loss
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # EMA update
            if self.ema:
                self.ema.update(self.model)
            
            # Metrics
            with torch.no_grad():
                if move_indices is not None:
                    pred_moves = policy_logits.argmax(dim=1)
                    policy_acc = (pred_moves == move_indices).float().mean()
                else:
                    policy_acc = torch.tensor(0.0)
                
                value_acc = (torch.abs(torch.tanh(pred_value) - results) < 0.2).float().mean()
            
            self.metrics.update(
                loss=loss,
                value_loss=value_loss,
                policy_loss=policy_loss,
                policy_acc=policy_acc,
                value_acc=value_acc
            )
            
            # Log
            if batch_idx % 10 == 0:
                print(f"[Epoch {self.current_epoch}] [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {self.metrics.avg('loss'):.4f} | "
                      f"V-Loss: {self.metrics.avg('value_loss'):.4f} | "
                      f"P-Loss: {self.metrics.avg('policy_loss'):.4f}", end='\r')
        
        print()  # New line after progress
        return self.metrics.average()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.metrics.reset()
        
        for batch in self.val_loader:
            # Unpack
            if len(batch) == 3:
                boards, results, move_indices = batch
            elif len(batch) == 2:
                boards, results = batch
                move_indices = None
            else:
                continue
            
            boards = boards.to(self.device)
            results = results.to(self.device).float().unsqueeze(1)
            if move_indices is not None:
                move_indices = move_indices.to(self.device).long()
            
            with autocast(enabled=self.use_amp):
                pred_value, policy_logits = self.model(boards)
                
                value_loss = self.value_criterion(torch.tanh(pred_value), results)
                
                if move_indices is not None:
                    policy_loss = self.policy_criterion(policy_logits, move_indices)
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)
                
                loss = value_loss + 0.5 * policy_loss
            
            # Metrics
            if move_indices is not None:
                pred_moves = policy_logits.argmax(dim=1)
                policy_acc = (pred_moves == move_indices).float().mean()
            else:
                policy_acc = torch.tensor(0.0)
            
            value_acc = (torch.abs(torch.tanh(pred_value) - results) < 0.2).float().mean()
            
            self.metrics.update(
                val_loss=loss,
                val_value_loss=value_loss,
                val_policy_loss=policy_loss,
                val_policy_acc=policy_acc,
                val_value_acc=value_acc
            )
        
        return self.metrics.average()
    
    def train(self, epochs: int, early_stopping_patience: int = 15):
        """Main training loop"""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update history
            for key, value in all_metrics.items():
                self.history[key].append(value)
            
            # Print summary
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train: Loss={train_metrics.get('loss', 0):.4f} | "
                  f"V-Loss={train_metrics.get('value_loss', 0):.4f} | "
                  f"P-Loss={train_metrics.get('policy_loss', 0):.4f}")
            if val_metrics:
                print(f"Val:   Loss={val_metrics.get('val_loss', 0):.4f} | "
                      f"V-Loss={val_metrics.get('val_value_loss', 0):.4f} | "
                      f"P-Loss={val_metrics.get('val_policy_loss', 0):.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}\n")
            
            # Save best
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save("caro_best.pt")
                print(f"💾 New best model! Val loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save(f"caro_epoch{epoch}.pt")
        
        print("\n✅ Training complete!")
        return dict(self.history)
    
    def save(self, filename: str):
        """Save checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            metrics={'best_val_loss': self.best_val_loss},
            ema=self.ema
        )


# ═══════════════════════════════════════════════════════════════════════════
# ✅ CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    data_dir: str = "data/professional",
    epochs: int = 120,
    batch_size: int = 128,
    lr: float = 1e-4,
    use_amp: bool = True,
    use_ema: bool = True,
    early_stopping_patience: int = 30,
    **kwargs
):
    """Convenience function to train model"""
    
    # Create data loaders with FIXED dataset
    print("📂 Loading data...")
    full_dataset = CaroDataset(data_dir, use_augmentation=True)  # ✅ Enable augmentation
    
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Train: {train_size} | Val: {val_size}")
    
    # Build model
    model = build_preset("large")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        use_amp=use_amp,
        use_ema=use_ema,
        epochs=epochs,
        **kwargs
    )
    
    # Train
    history = trainer.train(epochs=epochs, early_stopping_patience=early_stopping_patience)
    
    print(f"\n💾 Best model saved to: checkpoints/caro_best.pt")
    return model, history


if __name__ == "__main__":
    train_model(
        data_dir="data/professional",
        epochs=120,
        batch_size=128,
        lr=1e-4,
        use_amp=True,
        use_ema=True,
        early_stopping_patience=30
    )