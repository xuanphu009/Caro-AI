
import sys
import os

from trainer import train_model

if __name__ == "__main__":
    model, history = train_model(
        data_dir="data/professional",
        model_preset="large",
        epochs=150,              # tang dan vi co du lieu nhieu hon
        batch_size=256,          # tang dan
        lr=1e-4,                # giảm learning rate một chút (ổn định hơn)
        use_ema=True,
        use_amp = True,
        early_stopping_patience= 30  # neu 20 epochs khong cai thien thi dung
    )
    print("Training complete!")