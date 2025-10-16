# src/train_cnn.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trainer import train_model

if __name__ == "__main__":
    model, history = train_model(
        data_dir="data/professional",
        model_preset="large",
        epochs=120,              # tang dan vi co du lieu nhieu hon
        batch_size=64,          # tang dan
        lr=5e-4,                # giảm learning rate một chút (ổn định hơn)
        use_ema=True,
        early_stopping_patience=1000  # Tắt early stopping
    )
    print("Training complete!")