"""
Create a reduced-class YOLO dataset (images + labels + data.yaml) from an existing one.

Example:
    python filter_yolo_classes.py \
        --src yolo_sld \
        --dst yolo_sld_2 \
        --classes transformer breaker \
        --keep-empty

Notes:
- Copies images/labels for each split and remaps class ids to a new contiguous range.
- By default, drops images that end up with zero boxes after filtering; use --keep-empty
  if you want to retain them as background-only images.
- Avoids PyYAML dependency by parsing names from data.yaml manually.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="yolo_sld", help="Source YOLO dataset root")
    ap.add_argument("--dst", type=str, default="yolo_sld_2", help="Destination root for filtered dataset")
    ap.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Class names to keep (ordered; defines new id mapping)",
    )
    ap.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep images that lose all boxes after filtering",
    )
    ap.add_argument(
        "--clear",
        action="store_true",
        help="Remove destination folder if it exists",
    )
    return ap.parse_args()


def load_names(data_yaml: Path) -> List[str]:
    """Parse class names from a simple YOLO data.yaml without requiring PyYAML."""
    names = []
    recording = False
    with open(data_yaml, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("names"):
                recording = True
                continue
            if recording:
                if ":" in line:
                    try:
                        idx_str, name = line.split(":", 1)
                        idx = int(idx_str.strip())
                        names.append((idx, name.strip()))
                        continue
                    except ValueError:
                        break
                else:
                    break
    # Sort by idx
    return [name for _, name in sorted(names, key=lambda x: x[0])]


def ensure_dirs(dst: Path, splits: List[str], clear: bool):
    if dst.exists():
        if clear:
            shutil.rmtree(dst)
        else:
            raise SystemExit(f"Destination exists: {dst}. Use --clear to overwrite.")
    for split in splits:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_filtered(
    split: str,
    src_img: Path,
    src_lbl: Path,
    dst_img: Path,
    dst_lbl: Path,
    id_map: Dict[int, int],
    keep_empty: bool,
) -> Tuple[int, int]:
    """Return (images_kept, boxes_kept)."""
    if not src_img.exists():
        return 0, 0

    kept_images = 0
    kept_boxes = 0

    images = sorted(
        [p for p in src_img.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    for img_path in images:
        lbl_path = src_lbl / f"{img_path.stem}.txt"
        new_lines = []

        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        old_cls = int(parts[0])
                    except ValueError:
                        continue
                    if old_cls not in id_map:
                        continue
                    new_cls = id_map[old_cls]
                    coords = parts[1:5]
                    new_lines.append(f"{new_cls} " + " ".join(coords))

        if not new_lines and not keep_empty:
            continue

        # Copy image + write filtered label
        shutil.copy2(img_path, dst_img / img_path.name)
        with open(dst_lbl / f"{img_path.stem}.txt", "w") as f:
            for ln in new_lines:
                f.write(ln + "\n")

        kept_images += 1
        kept_boxes += len(new_lines)

    return kept_images, kept_boxes


def write_data_yaml(dst: Path, names: List[str]):
    yaml_lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for idx, name in enumerate(names):
        yaml_lines.append(f"  {idx}: {name}")
    (dst / "data.yaml").write_text("\n".join(yaml_lines) + "\n")


def main():
    args = parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    data_yaml = src / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Missing source data.yaml: {data_yaml}")

    src_names = load_names(data_yaml)
    name_to_idx = {n: i for i, n in enumerate(src_names)}

    # Validate requested classes
    missing = [c for c in args.classes if c not in name_to_idx]
    if missing:
        raise SystemExit(f"Requested classes not in source dataset: {missing}")

    id_map = {name_to_idx[c]: i for i, c in enumerate(args.classes)}

    splits = ["train", "val", "test"]
    ensure_dirs(dst, splits, args.clear)

    total_images = 0
    total_boxes = 0
    for split in splits:
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        dst_img = dst / "images" / split
        dst_lbl = dst / "labels" / split

        kept_i, kept_b = copy_filtered(
            split, src_img, src_lbl, dst_img, dst_lbl, id_map, args.keep_empty
        )
        total_images += kept_i
        total_boxes += kept_b
        print(f"[{split}] kept images={kept_i} boxes={kept_b}")

    write_data_yaml(dst, args.classes)
    print(f"[DONE] Wrote filtered dataset → {dst}")
    print(f"Total images: {total_images}, boxes: {total_boxes}")


if __name__ == "__main__":
    main()
