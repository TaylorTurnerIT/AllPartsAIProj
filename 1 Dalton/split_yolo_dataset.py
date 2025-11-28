"""
Split the YOLO dataset into train/val/test (images + labels) without leakage.

Usage:
    python split_yolo_dataset.py --val 0.15 --test 0.15 --seed 42

Notes:
- Works even if you already have data in train/val/test; it rebuilds splits
  from all available images and matching labels.
- Labels are moved alongside images; any missing label files are reported.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

SPLITS = ["train", "val", "test"]


def build_label_index(lbl_root: Path) -> Dict[str, Path]:
    """Return stem -> label path across all current label splits."""
    index: Dict[str, Path] = {}
    for split in SPLITS:
        ldir = lbl_root / split
        if not ldir.exists():
            continue
        for lbl in ldir.glob("*.txt"):
            index[lbl.stem] = lbl
    return index


def collect_images(img_root: Path) -> List[Path]:
    """Gather all images from existing train/val/test folders."""
    imgs: List[Path] = []
    for split in SPLITS:
        idir = img_root / split
        if not idir.exists():
            continue
        imgs.extend(sorted(idir.glob("*.png")))
        imgs.extend(sorted(idir.glob("*.jpg")))
        imgs.extend(sorted(idir.glob("*.jpeg")))
    return imgs


def main(root: Path, val_ratio: float, test_ratio: float, seed: int):
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val + test ratio must be < 1.0")

    img_root = root / "images"
    lbl_root = root / "labels"

    all_imgs = collect_images(img_root)
    if not all_imgs:
        raise SystemExit(f"No images found under {img_root}/* to split.")

    label_index = build_label_index(lbl_root)

    random.seed(seed)
    random.shuffle(all_imgs)

    n = len(all_imgs)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    val_imgs = all_imgs[:n_val]
    test_imgs = all_imgs[n_val : n_val + n_test]
    train_imgs = all_imgs[n_val + n_test :]

    # Temporary staging area
    tmp_root = root / "_split_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_img_root = tmp_root / "images"
    tmp_lbl_root = tmp_root / "labels"

    for split in SPLITS:
        (tmp_img_root / split).mkdir(parents=True, exist_ok=True)
        (tmp_lbl_root / split).mkdir(parents=True, exist_ok=True)

    def place(img_path: Path, split: str):
        dest_img = tmp_img_root / split / img_path.name
        dest_lbl = tmp_lbl_root / split / f"{img_path.stem}.txt"

        # Copy image
        shutil.copy2(img_path, dest_img)

        lbl_path = label_index.get(img_path.stem)
        if lbl_path and lbl_path.exists():
            shutil.copy2(lbl_path, dest_lbl)
        else:
            print(f"[WARN] Missing label for {img_path.name}")

    for img in train_imgs:
        place(img, "train")
    for img in val_imgs:
        place(img, "val")
    for img in test_imgs:
        place(img, "test")

    # Replace old splits with the staged ones
    for split in SPLITS:
        idir = img_root / split
        ldir = lbl_root / split
        if idir.exists():
            shutil.rmtree(idir)
        if ldir.exists():
            shutil.rmtree(ldir)
        shutil.move(str(tmp_img_root / split), str(idir))
        shutil.move(str(tmp_lbl_root / split), str(ldir))

    shutil.rmtree(tmp_root)

    # Remove stale YOLO caches if present
    for cache in lbl_root.glob("*.cache"):
        cache.unlink()
    for cache in (lbl_root / "train").glob("*.cache"):
        cache.unlink()

    print(
        f"[DONE] Split {n} images → "
        f"{len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="yolo_sld", help="Dataset root containing images/ and labels/")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(root=Path(args.root), val_ratio=args.val, test_ratio=args.test, seed=args.seed)
