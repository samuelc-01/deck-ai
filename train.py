import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from imageFolder import train_loader, val_loader
from imageFolder import train_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
# Fix: use len(train_data.classes) for output features
model.fc = nn.Linear(num_features, len(train_data.classes))
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%\n")

    # Save the trained model
    torch.save(model.state_dict(), "resnet18_card_naipes.pth")
    print("Modelo salvo em 'resnet18_card_naipes.pth'")
