from dataset import CaroDataset
from trainer import train_model
from model import build_preset

dataset = CaroDataset(data_dir="data/selfplay")
print("Dataset:", len(dataset))

model = build_preset("xlarge")  # hoặc "small", "base", "large", "xlarge"
train_model(model, dataset, epochs=50, batch_size=128, lr=1e-3) # 10->50, 64->128, tiny->xlarge
