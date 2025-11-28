import os
import shutil
from pathlib import Path

SRC = Path("dataset/processed")     # contains /png and /aug subfolders
DEST = Path("dataset/classes")

DEST.mkdir(exist_ok=True)

for file in SRC.rglob("*.png"):     # <-- FIXED: recursive search
    name = file.stem  # e.g. "42_4-3-center-closed-lever-spring-return.e62eeefbfb"
    
    parts = name.split("_", 1)
    if len(parts) < 2:
        print(f"[SKIP] no numeric prefix: {file.name}")
        continue

    label_raw = parts[1]
    label = label_raw.split(".", 1)[0]  # remove hash suffix

    dest_folder = DEST / label
    dest_folder.mkdir(exist_ok=True)

    shutil.copy(file, dest_folder / file.name)
    print(f"[COPY] {file.name} → {dest_folder}")