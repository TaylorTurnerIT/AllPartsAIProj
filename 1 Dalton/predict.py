import argparse
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "symbol_classifier.pth"
CLASSES_PATH = "classes.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --------------------------
# Load class labels
# --------------------------
with open(CLASSES_PATH, "r") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

# --------------------------
# Image transform
# --------------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_model():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, device


def predict(image_path):
    model, device = load_model()

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]

    # Top-5 predictions
    top5_prob, top5_idx = torch.topk(probs, 5)

    print("\n=== Predictions ===")
    for p, idx in zip(top5_prob, top5_idx):
        print(f"{idx_to_class[idx.item()]:60s}  |  {p.item():.4f}")

    best_idx = top5_idx[0].item()
    best_class = idx_to_class[best_idx]
    best_prob = top5_prob[0].item()

    print("\nTop-1 Prediction:")
    print(f"{best_class}  ({best_prob:.4f})")

    return best_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an image")
    args = parser.parse_args()

    predict(args.image)
