"""
Quickly inspect YOLO label balance and box sizes across splits.

Usage:
    python analyze_yolo_labels.py --data yolo_sld/data.yaml

Outputs:
    - Images per split
    - Box counts per split
    - Per-class counts and % share
    - Suggested inverse-frequency weights (normalized)
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="yolo_sld/data.yaml", help="Path to data.yaml")
    return ap.parse_args()


def load_names(data_yaml: Path):
    """Parse class names from a simple YOLO data.yaml (robust to missing PyYAML)."""
    try:
        import yaml  # type: ignore

        with open(data_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names")
        if isinstance(names, dict):
            # dict of idx: name
            return [names[k] for k in sorted(names.keys())]
        return list(names)
    except Exception:
        # Fallback: naive parser for "  0: name" lines
        names = []
        recording = False
        with open(data_yaml, "r") as f:
            for line in f:
                if line.strip().startswith("names"):
                    recording = True
                    continue
                if recording:
                    if ":" in line:
                        parts = line.split(":", 1)
                        try:
                            idx = int(parts[0].strip())
                            name = parts[1].strip()
                            names.append((idx, name))
                        except ValueError:
                            break
                    else:
                        break
        names = [n for _, n in sorted(names, key=lambda x: x[0])]
        return names


def gather_labels(lbl_dir: Path, num_classes: int):
    counts = Counter()
    box_area = defaultdict(list)
    total_images = 0
    total_boxes = 0

    if not lbl_dir.exists():
        return counts, box_area, total_images, total_boxes

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        total_images += 1
        if lbl_file.name.endswith(".cache"):
            continue
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue
                if cls < 0 or cls >= num_classes:
                    continue
                counts[cls] += 1
                total_boxes += 1
                box_area[cls].append(w * h)

    return counts, box_area, total_images, total_boxes


def main():
    args = parse_args()
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    names = load_names(data_yaml)
    num_classes = len(names)

    root = data_yaml.parent
    splits = ["train", "val", "test"]

    split_info = {}
    class_totals = Counter()
    class_area = defaultdict(list)

    for split in splits:
        lbl_dir = root / "labels" / split
        counts, areas, n_img, n_box = gather_labels(lbl_dir, num_classes)
        split_info[split] = {"images": n_img, "boxes": n_box, "counts": counts}
        class_totals.update(counts)
        for k, v in areas.items():
            class_area[k].extend(v)

    total_boxes = sum(class_totals.values())
    print(f"Loaded classes: {names}")
    for split in splits:
        info = split_info[split]
        print(
            f"{split:5s}: images={info['images']:5d} boxes={info['boxes']:6d}"
        )

    print("\nPer-class counts:")
    for cls_idx, name in enumerate(names):
        cnt = class_totals.get(cls_idx, 0)
        share = (cnt / total_boxes * 100) if total_boxes else 0
        areas = class_area.get(cls_idx, [])
        avg_area = sum(areas) / len(areas) if areas else 0.0
        print(
            f"{cls_idx:2d} {name:20s} count={cnt:6d} share={share:5.2f}% avg_box_area={avg_area:.4f}"
        )

    if total_boxes:
        print("\nSuggested inverse-frequency weights (normalized):")
        inv = {i: 1.0 / max(class_totals.get(i, 1), 1) for i in range(num_classes)}
        norm = sum(inv.values())
        weights = {names[i]: inv[i] / norm for i in range(num_classes)}
        print(json.dumps(weights, indent=2))


if __name__ == "__main__":
    main()
