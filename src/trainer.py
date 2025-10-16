# trainer.py

import os
import sys
import time
import math
import json
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import CaroNet, build_preset, save_checkpoint, load_checkpoint, EMA
from dataset import CaroDataset

# CONFIGURATION
class TrainingConfig:
    # Cấu hình huấn luyện với các giá trị mặc định hợp lý
    
    data_dir: str = "data/professional"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    model_preset: str = "medium"
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.1
    
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    label_smoothing: float = 0.1
    dropout: float = 0.1
    gradient_clip: float = 1.0
    
    use_amp: bool = False  # Disable on CPU by default
    use_ema: bool = True
    ema_decay: float = 0.999
    
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 0.5
    
    early_stopping_patience: int = 15
    
    log_interval: int = 10
    val_interval: int = 1
    save_interval: int = 5
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# METRICS TRACKER
class MetricsTracker:
    # Theo dõi và tính trung bình các chỉ số huấn luyện

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.counts = {}

    def update(self, **kwargs):
        # Update metrics with batch values
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
        # Return average of given metric key
        if key not in self.metrics:
            return 0.0
        return self.metrics[key] / max(1, self.counts[key])

    def summary(self):
        #Return a summary dict of all averages
        return {k: self.avg(k) for k in self.metrics}

    def __str__(self):
        parts = []
        for k, v in self.summary().items():
            parts.append(f"{k}: {v:.4f}")
        return " | ".join(parts)

# TRAINER CLASS
class Trainer:
    #Training engine for Caro CNN
    
    def __init__(self, config: Optional[TrainingConfig] = None, **kwargs):
        """
        Initialize trainer
        
        Args:
            config: TrainingConfig object
            **kwargs: Override config parameters
        """
        self.config = config or TrainingConfig()
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Setup directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Device
        self.device = torch.device(self.config.device)
        print(f"Using device: {self.device}")
        
        # Model
        print(f"Building {self.config.model_preset} model...")
        self.model = build_preset(self.config.model_preset, dropout=self.config.dropout)
        self.model.to(self.device)
        
        # Data loaders (will be set later)
        self.train_loader = None
        self.val_loader = None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = None
        
        # Mixed precision
        self.config.use_amp = self.config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.config.use_amp)
        
        # EMA
        self.ema = EMA(self.model, decay=self.config.ema_decay) if self.config.use_ema else None
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        # Metrics
        self.metrics = MetricsTracker()
        self.history = defaultdict(list)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        print("Trainer initialized")
    
    def setup_data(self, data_dir: str = None):
        # Setup data loaders
        if data_dir is None:
            data_dir = self.config.data_dir
        
        print(f"\nLoading data from {data_dir}...")
        
        # Load dataset
        dataset = CaroDataset(data_dir, min_game_length=5)
        
        if len(dataset) == 0:
            raise ValueError(f"No data found in {data_dir}")
        
        print(f"Loaded {len(dataset)} positions")
        
        # Split into train/val
        val_size = int(len(dataset) * self.config.val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        print(f"Train: {train_size} | Val: {val_size}")
        
        # Create loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Setup scheduler
        self._build_scheduler()
    
    def _build_scheduler(self):
        #Build learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.min_lr
        )
    
    def train_epoch(self) -> Dict[str, float]:
        # Train one epoch
        self.model.train()
        self.metrics.reset()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack batch (handle 2 or 3 outputs)
            if len(batch) == 3:
                boards, results, move_indices = batch
            elif len(batch) == 2:
                boards, results = batch
                move_indices = None
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)}")
            
            boards = boards.to(self.device)
            results = results.to(self.device).float().unsqueeze(1)
            if move_indices is not None:
                move_indices = move_indices.to(self.device).long()
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.use_amp):
                pred_value, policy_logits = self.model(boards)
                
                # Losses
                value_loss = self.value_criterion(torch.tanh(pred_value), results)
                
                if move_indices is not None:
                    policy_loss = self.policy_criterion(policy_logits, move_indices)
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)
                
                loss = (self.config.value_loss_weight * value_loss + 
                       self.config.policy_loss_weight * policy_loss)
            
            # Backward
            self.scaler.scale(loss).backward()
            
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
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
                
                value_pred = torch.tanh(pred_value)
                value_acc = (torch.abs(value_pred - results) < 0.5).float().mean()
            
            self.metrics.update(
                loss=loss,
                value_loss=value_loss,
                policy_loss=policy_loss,
                policy_acc=policy_acc,
                value_acc=value_acc
            )
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                print(f"[Epoch {self.current_epoch}] [{batch_idx}/{len(self.train_loader)}] {self.metrics}", end='\r')
            
            self.global_step += 1
        
        return self.metrics.summary()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        # Validate on validation set
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.metrics.reset()
        
        for batch in self.val_loader:
            # Unpack batch
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
            
            with autocast(enabled=self.config.use_amp):
                pred_value, policy_logits = self.model(boards)
                
                value_loss = self.value_criterion(torch.tanh(pred_value), results)
                if move_indices is not None:
                    policy_loss = self.policy_criterion(policy_logits, move_indices)
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)
                
                loss = (self.config.value_loss_weight * value_loss + 
                       self.config.policy_loss_weight * policy_loss)
            
            # Metrics
            if move_indices is not None:
                pred_moves = policy_logits.argmax(dim=1)
                policy_acc = (pred_moves == move_indices).float().mean()
            else:
                policy_acc = torch.tensor(0.0, device=self.device)
            
            value_pred = torch.tanh(pred_value)
            value_acc = (torch.abs(value_pred - results) < 0.5).float().mean()
            
            self.metrics.update(
                val_loss=loss,
                val_value_loss=value_loss,
                val_policy_loss=policy_loss,
                val_policy_acc=policy_acc,
                val_value_acc=value_acc
            )
        
        return self.metrics.summary()
    
    def train(self, epochs: Optional[int] = None):
        # Main training loop
        if epochs is None:
            epochs = self.config.epochs
        
        if self.train_loader is None:
            raise ValueError("Data not loaded. Call setup_data() first.")
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = {}
            if epoch % self.config.val_interval == 0:
                val_metrics = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update history
            for key, value in all_metrics.items():
                self.history[key].append(value)
            
            # Print summary
            epoch_time = time.time() - epoch_start
            self._print_epoch(epoch, all_metrics, epoch_time)
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save(f"caro_epoch{epoch}.pt")
            
            # Early stopping
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save("caro_best.pt", is_best=True)
                print(f"New best model! Val loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                if epoch > self.config.warmup_epochs + 10:
                    if self.epochs_without_improvement >= self.config.early_stopping_patience:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        return dict(self.history)
    
    def _print_epoch(self, epoch, metrics, elapsed):
        # Print epoch summary
        print(f"\nEpoch {epoch} ({elapsed:.1f}s)")
        print(f"Train Loss: {metrics.get('loss', 0):.4f}")
        if 'val_loss' in metrics:
            print(f"Val Loss:   {metrics['val_loss']:.4f}")
        print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def save(self, filename: str, is_best: bool = False):
        # Save checkpoint
        path = os.path.join(self.config.checkpoint_dir, filename)
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            ema=self.ema
        )
        if is_best:
            print(f"Saved best model → {path}")

# TRAINING FUNCTION
def train_model(
    data_dir: str = "data/professional",
    model_preset: str = "large",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 5e-4,
    **kwargs
) -> Tuple[CaroNet, Dict]:
    """
    Convenience function to train model
    
    Args:
        data_dir: Data directory
        model_preset: Model size
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        **kwargs: Additional config
    
    Returns:
        model, history
    """
    # Create trainer
    trainer = Trainer(
        data_dir=data_dir,
        model_preset=model_preset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        **kwargs
    )
    
    # Setup data
    trainer.setup_data(data_dir)
    
    # Train
    history = trainer.train(epochs=epochs)
    
    return trainer.model, history
