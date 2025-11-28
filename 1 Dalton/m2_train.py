import torch
from ultralytics import YOLO

# ------------------------------
# DEVICE CHECK (Mac GPU / MPS)
# ------------------------------
print("MPS available:", torch.backends.mps.is_available())
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# ------------------------------
# SETTINGS (Tuned for M2 Air 24GB)
# ------------------------------
MODEL_PATH = "yolo11n.pt"           # v8 example

DATA_PATH = "yolo_sld_2/data.yaml"             # Your dataset YAML

# ------------------------------
# LOAD MODEL
# ------------------------------
model = YOLO(MODEL_PATH)

# ------------------------------
# TRAIN
# ------------------------------
results = model.train(
    data=DATA_PATH,
    epochs=50,
    imgsz=640,
    batch=2,
    fraction=0.5,
    device="mps",
    workers=1,
    half=False,

    # TURN OFF HEAVY AUGMENTATIONS
    mosaic=0.5,
    auto_augment=False,
    erasing=0.2,
    hsv_s=0.4,
    hsv_v=0.2,
    fliplr=0.5,

    # TRAINING SPEED
    cos_lr=True,
    optimizer="Adam",
    lr0=0.001,
    val=False
)

print("Training complete.")