import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATASET_DIR = "dataset/splits"
BATCH = 16
EPOCHS = 10
LR = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print("Using device:", device)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# Load datasets
train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform)
val_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform)
test_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)
test_loader = DataLoader(test_ds, batch_size=BATCH)

# Load pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def run_eval(loader, split_name):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1) * 100
    return avg_loss, acc


# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / len(train_ds)
    val_loss, val_acc = run_eval(val_loader, "val")

    print(
        f"[Epoch {epoch+1}/{EPOCHS}] "
        f"train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.2f}%"
    )

print("Training complete!")

# Final test evaluation
test_loss, test_acc = run_eval(test_loader, "test")
print(f"[Test] loss={test_loss:.4f} acc={test_acc:.2f}%")

torch.save(model.state_dict(), "symbol_classifier.pth")
print("Saved model → symbol_classifier.pth")
