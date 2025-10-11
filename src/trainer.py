# huấn luyện model từ dataset


# src/trainer.py
"""
Trainer for SimpleCaroNet model (from model.py)
Train both value + policy heads using self-play dataset (JSON -> dataset.py).

Requirements:
  - dataset.py defines CaroDataset returning (board, result, move_idx)
  - model.py defines SimpleCaroNet + build_model + save_checkpoint
  - data stored in data/processed/*.json

Features:
  - Mixed Precision (AMP) if CUDA available
  - Periodic checkpoint saving
  - Automatic learning rate scheduler
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from model import build_model, save_checkpoint
from dataset import CaroDataset

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/processed/"
SAVE_DIR = "checkpoints/"
BATCH_SIZE = 128
EPOCHS = 80
LR = 1e-3
SAVE_EVERY = 10

# -------------------------------------------------------------
# TRAIN FUNCTION
# -------------------------------------------------------------
def train_model(
    data_dir=DATA_DIR,
    save_dir=SAVE_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    save_every=SAVE_EVERY,
    base_channels=64,
    n_blocks=6,
):
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Load dataset ---
    dataset = CaroDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"✅ Loaded {len(dataset)} samples from {data_dir}")

    # --- 2. Build model ---
    model = build_model(base_channels=base_channels, n_blocks=n_blocks).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # --- 3. Training loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0

        for boards, results, move_idx in dataloader:
            boards = boards.to(DEVICE)                    # (B,2,15,15)
            results = results.to(DEVICE).float().unsqueeze(1)  # (B,1)
            move_idx = move_idx.to(DEVICE).long()              # (B,)

            optimizer.zero_grad()

            with autocast(enabled=(DEVICE == "cuda")):
                pred_value, policy_logits = model(boards)

                value_loss = F.mse_loss(torch.tanh(pred_value), results)
                policy_loss = F.cross_entropy(policy_logits, move_idx)
                loss = value_loss + 0.5 * policy_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()

        scheduler.step()

        print(f"[Epoch {epoch:03d}] "
              f"Loss={total_loss/len(dataloader):.4f} "
              f"Value={total_value_loss/len(dataloader):.4f} "
              f"Policy={total_policy_loss/len(dataloader):.4f} "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        # --- 4. Save checkpoint periodically ---
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(save_dir, f"caro_epoch{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch)
            print(f"💾 Saved checkpoint → {ckpt_path}")

    print("✅ Training finished.")
    return model


train_model()
