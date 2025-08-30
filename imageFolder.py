from torchvision import datasets
from torch.utils.data import DataLoader
from transforms_config import transform

train_data = datasets.ImageFolder("archive/train", transform=transform)
val_data = datasets.ImageFolder("archive/valid", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

print("Classes:", train_data.classes)
