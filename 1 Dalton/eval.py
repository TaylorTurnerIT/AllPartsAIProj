import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

DATASET_DIR = "dataset/splits"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
MODEL_PATH = "symbol_classifier.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print("Using device:", DEVICE)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

test_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(test_ds.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=test_ds.classes))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(all_labels, all_preds))
