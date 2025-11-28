import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC = Path("dataset/classes")
OUT = Path("dataset/splits")

# Directories to create
for d in ["train", "val", "test"]:
    subset_dir = OUT / d
    if subset_dir.exists():
        # clear previous contents to avoid stale files leaking across reruns
        for item in subset_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
    subset_dir.mkdir(parents=True, exist_ok=True)

# Folders that are NOT real symbol classes
BAD = {"1024"}  # add more if needed

for cls in SRC.iterdir():
    if not cls.is_dir():
        continue
    if cls.name in BAD:
        print(f"[SKIP] {cls.name} (not a real class)")
        continue

    images = list(cls.glob("*.png"))

    if len(images) < 5:
        print(f"[SKIP] {cls.name} (too few images: {len(images)})")
        continue

    # 10% test
    train_imgs, test_imgs = train_test_split(images, test_size=0.10, random_state=42)

    # 10% of remaining for validation
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.10, random_state=42)

    # Copy files to split folders
    for subset, name in [
        (train_imgs, "train"),
        (val_imgs, "val"),
        (test_imgs, "test"),
    ]:
        dest = OUT / name / cls.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy(img, dest / img.name)

    print(
        f"[OK] {cls.name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test"
    )
