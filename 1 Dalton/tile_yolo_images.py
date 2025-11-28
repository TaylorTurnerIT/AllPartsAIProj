"""
Tile large YOLO images into smaller patches and remap labels.

This is helpful when SLD sheets are very large (e.g., 7000x5000) and symbols are tiny.

Example:
    python tile_yolo_images.py --root yolo_sld --out yolo_sld_tiled \\
        --tile 1024 1024 --stride 1024 1024 --min-frac 0.1 --keep-empty
"""

import argparse
import math
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="yolo_sld", help="Dataset root (contains images/, labels/)")
    ap.add_argument("--out", type=str, default="yolo_sld_tiled", help="Output root for tiled dataset")
    ap.add_argument("--tile", nargs=2, type=int, metavar=("W", "H"), default=[1024, 1024], help="Tile width height")
    ap.add_argument("--stride", nargs=2, type=int, metavar=("SX", "SY"), default=None, help="Stride (defaults to tile size)")
    ap.add_argument("--min-frac", type=float, default=0.1, help="Minimum fraction of original box area to keep after clipping")
    ap.add_argument("--keep-empty", action="store_true", help="Keep tiles even if they have no labels")
    ap.add_argument("--clear", action="store_true", help="Remove output root if it already exists")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Splits to process")
    return ap.parse_args()


def ensure_out_dirs(out_root: Path, splits: List[str]):
    if out_root.exists():
        if not any(out_root.iterdir()):
            pass
        elif args.clear:
            shutil.rmtree(out_root)
        else:
            raise SystemExit(f"{out_root} exists. Use --clear to overwrite.")
    for split in splits:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def yolo_to_xyxy(cx, cy, w, h, W, H):
    bw = w * W
    bh = h * H
    x1 = cx * W - bw / 2
    y1 = cy * H - bh / 2
    x2 = cx * W + bw / 2
    y2 = cy * H + bh / 2
    return x1, y1, x2, y2


def clip_box_to_tile(box: Tuple[float, float, float, float], tile):
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile
    cx1 = max(x1, tx1)
    cy1 = max(y1, ty1)
    cx2 = min(x2, tx2)
    cy2 = min(y2, ty2)
    return cx1, cy1, cx2, cy2


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def process_image(img_path: Path, lbl_path: Path, out_img_dir: Path, out_lbl_dir: Path, tile_w: int, tile_h: int, stride_w: int, stride_h: int, min_frac: float, keep_empty: bool):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    labels = []
    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, W, H)
                labels.append((cls, x1, y1, x2, y2))

    tile_count = 0
    for ty in range(0, H, stride_h):
        for tx in range(0, W, stride_w):
            tile_box = (tx, ty, min(tx + tile_w, W), min(ty + tile_h, H))
            tb_w = tile_box[2] - tile_box[0]
            tb_h = tile_box[3] - tile_box[1]
            if tb_w <= 0 or tb_h <= 0:
                continue

            new_labels = []
            for cls, x1, y1, x2, y2 in labels:
                clipped = clip_box_to_tile((x1, y1, x2, y2), tile_box)
                inter_area = box_area(clipped)
                orig_area = box_area((x1, y1, x2, y2))
                if orig_area <= 0 or inter_area <= 0:
                    continue
                if inter_area / orig_area < min_frac:
                    continue

                cx = ((clipped[0] + clipped[2]) / 2 - tile_box[0]) / tb_w
                cy = ((clipped[1] + clipped[3]) / 2 - tile_box[1]) / tb_h
                bw = (clipped[2] - clipped[0]) / tb_w
                bh = (clipped[3] - clipped[1]) / tb_h
                new_labels.append((cls, cx, cy, bw, bh))

            if not new_labels and not keep_empty:
                continue

            out_name = f"{img_path.stem}_x{tx}_y{ty}{img_path.suffix}"
            out_img_path = out_img_dir / out_name
            out_lbl_path = out_lbl_dir / f"{img_path.stem}_x{tx}_y{ty}.txt"

            img.crop(tile_box).save(out_img_path)
            with open(out_lbl_path, "w") as f:
                for cls, cx, cy, bw, bh in new_labels:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            tile_count += 1
    return tile_count


def main():
    global args
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out)

    tile_w, tile_h = args.tile
    stride_w, stride_h = args.stride if args.stride else args.tile

    ensure_out_dirs(out_root, args.splits)

    total_tiles = 0
    for split in args.splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        out_img_dir = out_root / "images" / split
        out_lbl_dir = out_root / "labels" / split

        if not img_dir.exists():
            print(f"[WARN] Missing split: {img_dir}")
            continue

        images = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )
        print(f"[{split}] found {len(images)} images")

        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            total_tiles += process_image(
                img_path,
                lbl_path,
                out_img_dir,
                out_lbl_dir,
                tile_w,
                tile_h,
                stride_w,
                stride_h,
                args.min_frac,
                args.keep_empty,
            )

    # Remove stale label caches if present
    for cache in out_root.glob("labels/*.cache"):
        cache.unlink()

    print(f"[DONE] Wrote {total_tiles} tiles to {out_root}")


if __name__ == "__main__":
    main()
