import os
import random
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CLASSES_JSON = "classes_9.json"
CLASS_DIR = Path("dataset/classes_9")
OUT_IMAGES = Path("yolo_sld/images/train")
OUT_LABELS = Path("yolo_sld/labels/train")
NUM_IMAGES = 6000     # Generate 6K training examples
MIN_SYMBOLS = 5
MAX_SYMBOLS = 25
CANVAS_SIZES = [(1024, 1024), (1280, 720), (720, 1280)]
ROTATION_ANGLES = [-45, -30, -15, 0, 15, 30, 45]
# Optional: up-weight rare classes (e.g., {"bus_tie_breaker": 3.0, "load_arrow": 2.0})
CLASS_SAMPLING_WEIGHTS = None

# Ensure output dirs exist
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

# Load class map
with open(CLASSES_JSON, "r") as f:
    CLASS_MAP = json.load(f)

CLASS_NAMES = list(CLASS_MAP.keys())
CLASS_NAME_TO_ID = CLASS_MAP

# -------------------------------------------------
# Load all class images into memory
# -------------------------------------------------
print("[*] Scanning symbol images...")
symbol_images = {}

for class_name in CLASS_NAMES:
    folder = CLASS_DIR / class_name
    if folder.exists():
        pngs = list(folder.glob("*.png"))
        if pngs:
            symbol_images[class_name] = pngs

print(f"[OK] Loaded {len(symbol_images)} symbol classes")

# Precompute class list + weights for sampling
sample_classes = list(symbol_images.keys())
sample_weights = None
if CLASS_SAMPLING_WEIGHTS:
    sample_weights = [CLASS_SAMPLING_WEIGHTS.get(cls, 1.0) for cls in sample_classes]

usage_counts = Counter()

# -------------------------------------------------
# Helper: paste symbol and return bounding box
# -------------------------------------------------
def paste_symbol(canvas, symbol_img, x, y, angle):
    rotated = symbol_img.rotate(angle, expand=True)
    w, h = rotated.size

    canvas.paste(rotated, (x, y), rotated)

    cx = x + w / 2
    cy = y + h / 2
    return cx, cy, w, h

# -------------------------------------------------
# Helper: random line noise
# -------------------------------------------------
def add_random_lines(draw, width, height, count=5):
    for _ in range(count):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        draw.line([(x1, y1), (x2, y2)],
                  fill=(0,0,0),
                  width=random.randint(1, 3))

# -------------------------------------------------
# MAIN synthetic generation
# -------------------------------------------------
print(f"[*] Generating {NUM_IMAGES} synthetic diagrams...")

for idx in range(NUM_IMAGES):

    W, H = random.choice(CANVAS_SIZES)
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    num_symbols = random.randint(MIN_SYMBOLS, MAX_SYMBOLS)
    labels = []

    chosen_classes = random.choices(sample_classes, weights=sample_weights, k=num_symbols)

    for cls in chosen_classes:
        symbol_path = random.choice(symbol_images[cls])
        symbol_img = Image.open(symbol_path).convert("RGBA")
        usage_counts[cls] += 1

        # ------------------------------
        # SAFE RESIZE BLOCK (Fixes crash)
        # ------------------------------
        # initial random scale
        scale = random.uniform(0.4, 1.5)
        new_w = int(symbol_img.width * scale)
        new_h = int(symbol_img.height * scale)

        # if symbol doesn't fit → shrink automatically
        if new_w >= W or new_h >= H:
            scale = min(W / symbol_img.width, H / symbol_img.height) * 0.8
            new_w = int(symbol_img.width * scale)
            new_h = int(symbol_img.height * scale)

        # skip if degenerate
        if new_w <= 0 or new_h <= 0:
            continue

        symbol_img = symbol_img.resize((new_w, new_h), Image.LANCZOS)

        # rotation
        angle = random.choice(ROTATION_ANGLES)
        rotated = symbol_img.rotate(angle, expand=True)

        rw, rh = rotated.size

        # final safety check
        if rw >= W or rh >= H:
            continue

        # ------------------------------
        # SAFE POSITION BLOCK (Fixes crash)
        # ------------------------------
        max_x = W - rw
        max_y = H - rh

        if max_x <= 0 or max_y <= 0:
            continue

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # paste & compute bbox
        canvas.paste(rotated, (x, y), rotated)

        cx = x + rw / 2
        cy = y + rh / 2

        labels.append((
            CLASS_NAME_TO_ID[cls],
            cx / W,
            cy / H,
            rw / W,
            rh / H
        ))

    # background noise
    add_random_lines(draw, W, H, count=random.randint(1, 8))

    # enhancements
    canvas = canvas.convert("RGB")
    canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.9, 1.1))
    canvas = canvas.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.0)))

    # save
    img_path = OUT_IMAGES / f"{idx:05}.png"
    canvas.save(img_path, "PNG")

    # labels
    txt_path = OUT_LABELS / f"{idx:05}.txt"
    with open(txt_path, "w") as f:
        for row in labels:
            f.write(" ".join(str(x) for x in row) + "\n")

    if idx % 100 == 0:
        print(f"[{idx}/{NUM_IMAGES}]")

print("[DONE] Synthetic dataset generated!")

# Simple histogram to verify sampling biases
print("\nClass usage histogram (symbols pasted):")
for cls in sorted(usage_counts.keys()):
    print(f"  {cls:20s} {usage_counts[cls]}")
