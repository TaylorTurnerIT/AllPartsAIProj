import os
import json

CLASSES_DIR = "dataset/classes"   # <-- FIXED

# Get class folders (ignore files like 1024 or other non-class dirs)
class_names = []

for name in os.listdir(CLASSES_DIR):
    full_path = os.path.join(CLASSES_DIR, name)
    if os.path.isdir(full_path):
        class_names.append(name)

class_names = sorted(class_names)

class_to_idx = {i: name for i, name in enumerate(class_names)}

with open("classes.json", "w") as f:
    json.dump(class_to_idx, f, indent=4)

print(f"[OK] Saved {len(class_to_idx)} classes → classes.json")