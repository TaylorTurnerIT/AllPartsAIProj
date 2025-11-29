"""
Run `new_best.pt` on an image with a two-pass trick:
 - Pass 1: original orientation (saves annotated image/labels)
 - Pass 2: 90° CCW rotated, boxes unrotated and merged to catch vertical breakers

Usage:
    python3 run_new_best.py [IMAGE_PATH] [--output JSON_PATH] [--inpainted INPAINTED_PATH]

    If no arguments provided, uses default bs.png
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Default inputs (used if no arguments provided)
DEFAULT_IMAGE_PATH = Path(__file__).parent / "bs.png"
DEFAULT_MODEL_PATH = Path(__file__).parent / "new_best.pt"
RUN_NAME = "new_best_manual"
IMG_SIZE = 1600
CONF = 0.25
IOU = 0.45
MERGE_IOU_BREAKER = 0.4  # IoU threshold to deduplicate breaker boxes
MERGE_IOU_TRANSFORMER = 0.45  # IoU threshold to deduplicate transformer boxes
ROTATED_CLASSES = {"breaker"}  # Only keep rotated-pass boxes for these classes
ROTATED_SUPPRESS_IOU = 0.5  # Skip rotated box if it overlaps a main-pass box this much or more
ROTATED_CONF_MIN = 0.4  # Minimum conf for rotated-pass boxes
ROTATED_TOPK = 30  # Keep at most this many rotated boxes after filtering
TRANSFORMER_CONF_MIN = 0.3
BREAKER_CONF_MIN = 0.25
KEEP_TOP_TRANSFORMERS = 2
KEEP_TOP_BREAKERS = 24
BOX_SHRINK_BREAKER = 0.85  # <1.0 shrinks breaker boxes
BOX_SHRINK_TRANSFORMER = 0.8  # <1.0 shrinks transformer boxes
TRANSFORMER_Y_MAX_FRAC = 0.7  # Drop transformer boxes centered below this normalized y
ANNOTATE_MERGED = True
INPAINT_ENABLED = True
INPAINT_PAD = 4  # Extra pixels around shrunk boxes when removing symbols
INPAINT_RADIUS = 3  # Radius for OpenCV inpaint (px)
INPAINT_CLASSES = {"breaker", "transformer"}


def map_box_back(angle: int, xyxy: Tuple[float, float, float, float], width: float, height: float) -> List[float]:
    """Map a box from a rotated image back to the original orientation."""
    x1, y1, x2, y2 = xyxy
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    if angle == 0:
        mapped = corners
    elif angle == 90:  # CCW
        mapped = [(width - y, x) for (x, y) in corners]
    elif angle == 270:  # CW
        mapped = [(y, height - x) for (x, y) in corners]
    elif angle == 180:
        mapped = [(width - x, height - y) for (x, y) in corners]
    else:
        raise ValueError(f"Unsupported angle: {angle}")

    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / max(area_a + area_b - inter_area, 1e-9)


def merge_boxes(boxes, iou_thresh_breaker=0.5, iou_thresh_transformer=0.35):
    """Greedy merge: keep highest-conf per class, drop boxes that overlap too much (class-specific IoU)."""
    boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    kept = []
    for b in boxes:
        cls = b["name"]
        thr = iou_thresh_breaker if cls == "breaker" else iou_thresh_transformer
        if any(b["cls"] == k["cls"] and iou(b["xyxy"], k["xyxy"]) >= thr for k in kept):
            continue
        kept.append(b)
    return kept


def shrink_box(xyxy: List[float], shrink: float, width: float, height: float) -> List[float]:
    """Shrink a box around its center by a factor, clipped to image bounds."""
    if shrink >= 1.0:
        return xyxy
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * shrink
    h = (y2 - y1) * shrink
    nx1 = max(0.0, cx - w / 2)
    ny1 = max(0.0, cy - h / 2)
    nx2 = min(width, cx + w / 2)
    ny2 = min(height, cy + h / 2)
    return [nx1, ny1, nx2, ny2]


def filter_boxes(boxes: List[dict], width: float, height: float) -> List[dict]:
    """Prune obviously bad boxes with simple heuristics (keeps breakers as-is)."""
    filtered = []
    for b in boxes:
        if b["name"] == "transformer":
            _, y1, _, y2 = b["xyxy"]
            y_center = ((y1 + y2) / 2) / height
            if y_center > TRANSFORMER_Y_MAX_FRAC:
                continue
        filtered.append(b)
    return filtered


def save_annotated(image_path: Path, boxes: List[dict], out_dir: Path):
    """Draw merged boxes on the original image for quick visual validation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        name = b["name"]
        conf = b["conf"]
        color = (0, 128, 255) if name == "transformer" else (0, 200, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{name} {conf:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        pad = 2
        label_y = max(0, y1 - text_h - pad * 2)
        draw.rectangle([x1, label_y, x1 + text_w + pad * 2, label_y + text_h + pad * 2], fill=color)
        draw.text((x1 + pad, label_y + pad), label, fill=(255, 255, 255), font=font)

    out_path = out_dir / f"{image_path.stem}_merged.jpg"
    img.save(out_path, quality=95)
    return out_path


def inpaint_symbols(image_path: Path, boxes: List[dict], out_dir: Path):
    """Remove detected symbols via inpainting to leave lines for tracing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image for inpaint: {image_path}")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for b in boxes:
        if INPAINT_CLASSES and b["name"] not in INPAINT_CLASSES:
            continue
        x1, y1, x2, y2 = b["xyxy"]
        x1 = max(0, int(round(x1 - INPAINT_PAD)))
        y1 = max(0, int(round(y1 - INPAINT_PAD)))
        x2 = min(img.shape[1] - 1, int(round(x2 + INPAINT_PAD)))
        y2 = min(img.shape[0] - 1, int(round(y2 + INPAINT_PAD)))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    inpainted = cv2.inpaint(img, mask, INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
    out_path = out_dir / f"{image_path.stem}_inpainted.png"
    cv2.imwrite(str(out_path), inpainted)
    return out_path


def save_merged_labels(boxes, width, height, out_dir: Path, stem: str):
    """Write YOLO txt with conf for merged boxes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / f"{stem}.txt"
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        x_c = (x1 + x2) / 2 / width
        y_c = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{b['cls']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {b['conf']:.6f}")
    label_path.write_text("\n".join(lines))
    return label_path


def main(image_path=None, output_json=None, output_inpainted=None, model_path=None):
    """
    Run detection with optional output paths.

    Args:
        image_path: Path to input image (default: bs.png)
        output_json: Path to save JSON output (optional)
        output_inpainted: Path to save inpainted image (optional)
        model_path: Path to model (default: new_best.pt)
    """
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))

    # Use defaults if not provided
    if image_path is None:
        image_path = DEFAULT_IMAGE_PATH
    else:
        image_path = Path(image_path)

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    else:
        model_path = Path(model_path)

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    model = YOLO(str(model_path))

    orientations = [
        ("main", 0, True),
        ("rot90", 90, False),
        ("rot270", 270, False),
    ]

    main_boxes = []
    rotated_boxes = []
    save_dir = Path("runs/detect") / RUN_NAME

    for tag, angle, do_save in orientations:
        if angle == 0:
            img_src = str(image_path)
        else:
            img_src = np.array(img.rotate(angle, expand=True))

        res = model.predict(
            source=img_src,
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            project="runs/detect",
            name=RUN_NAME if do_save else f"{RUN_NAME}_{tag}",
            exist_ok=True,
            save=do_save,
            save_txt=do_save,
            save_conf=do_save,
            device="cpu",
            verbose=do_save,
        )[0]

        if angle == 0 and hasattr(res, "save_dir"):
            save_dir = res.save_dir

        for box in res.boxes:
            cls_id = int(box.cls)
            cls_name = res.names[cls_id]
            conf = float(box.conf)
            xyxy_rot = [float(x) for x in box.xyxy[0]]
            xyxy = map_box_back(angle, xyxy_rot, width, height)

            if angle == 0:
                if cls_name == "transformer" and conf < TRANSFORMER_CONF_MIN:
                    continue
                if cls_name == "breaker" and conf < BREAKER_CONF_MIN:
                    continue
                main_boxes.append({"cls": cls_id, "name": cls_name, "conf": conf, "xyxy": xyxy})
            else:
                if ROTATED_CLASSES and cls_name not in ROTATED_CLASSES:
                    continue
                if conf < ROTATED_CONF_MIN:
                    continue
                if any(iou(xyxy, m["xyxy"]) >= ROTATED_SUPPRESS_IOU and cls_id == m["cls"] for m in main_boxes):
                    continue
                rotated_boxes.append({"cls": cls_id, "name": cls_name, "conf": conf, "xyxy": xyxy})
                if len(rotated_boxes) >= ROTATED_TOPK:
                    break

    merged = merge_boxes(
        main_boxes + rotated_boxes,
        iou_thresh_breaker=MERGE_IOU_BREAKER,
        iou_thresh_transformer=MERGE_IOU_TRANSFORMER,
    )
    merged = sorted(merged, key=lambda b: b["conf"], reverse=True)
    merged = filter_boxes(merged, width, height)

    # Trim to top-K per class
    keep = []
    breaker_count = 0
    transformer_count = 0
    for b in merged:
        if b["name"] == "breaker":
            if breaker_count >= KEEP_TOP_BREAKERS:
                continue
            breaker_count += 1
        elif b["name"] == "transformer":
            if transformer_count >= KEEP_TOP_TRANSFORMERS:
                continue
            transformer_count += 1
        keep.append(b)
    merged = keep

    # Shrink boxes (class-specific) and save
    shrunk = []
    for b in merged:
        shrink = BOX_SHRINK_BREAKER if b["name"] == "breaker" else BOX_SHRINK_TRANSFORMER
        shrunk_xyxy = shrink_box(b["xyxy"], shrink, width, height)
        shrunk.append({**b, "xyxy": shrunk_xyxy})

    merged_labels_dir = Path("runs/detect") / f"{RUN_NAME}_merged" / "labels"
    label_path = save_merged_labels(shrunk, width, height, merged_labels_dir, image_path.stem)
    merged_img_dir = merged_labels_dir.parent
    merged_img_path = None
    inpaint_img_path = None
    if ANNOTATE_MERGED:
        merged_img_path = save_annotated(image_path, shrunk, merged_img_dir)
    if INPAINT_ENABLED:
        # Use custom output path if provided, otherwise use default location
        if output_inpainted:
            inpaint_custom_path = Path(output_inpainted)
            inpaint_custom_path.parent.mkdir(parents=True, exist_ok=True)
            inpaint_img_path = inpaint_symbols(image_path, shrunk, inpaint_custom_path.parent)
            # Rename to the exact path requested
            if inpaint_img_path != inpaint_custom_path:
                import shutil
                shutil.move(str(inpaint_img_path), str(inpaint_custom_path))
                inpaint_img_path = inpaint_custom_path
        else:
            inpaint_img_path = inpaint_symbols(image_path, shrunk, merged_img_dir)

    print(f"\nSaved primary outputs to: {save_dir}")
    print(f"Merged labels (two-pass) saved to: {label_path}")
    if merged_img_path:
        print(f"Merged annotated image saved to: {merged_img_path}")
    if inpaint_img_path:
        print(f"Inpainted (symbols removed) image saved to: {inpaint_img_path}")
    print(f"Detections after merge: {len(shrunk)}")
    for i, b in enumerate(shrunk, 1):
        xyxy = [round(v, 1) for v in b["xyxy"]]
        print(f"{i}. {b['name']}  conf={b['conf']:.3f}  xyxy={xyxy}")

    # Generate JSON output if requested
    if output_json:
        symbols = []
        for idx, b in enumerate(shrunk):
            center = [
                (b["xyxy"][0] + b["xyxy"][2]) / 2,
                (b["xyxy"][1] + b["xyxy"][3]) / 2
            ]
            symbols.append({
                "id": idx,
                "cls_id": b["cls"],
                "name": b["name"],
                "conf": b["conf"],
                "bbox": b["xyxy"],
                "center": center
            })

        json_data = {
            "symbols": symbols,
            "lines": [],  # Will be filled by line detection
            "connections": []  # Will be filled by connection finding
        }

        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON output saved to: {output_json_path}")

    return shrunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLO detection with multi-orientation for better breaker detection"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to input image (default: bs.png in same directory)"
    )
    parser.add_argument(
        "--output",
        help="Path to save JSON output (optional)"
    )
    parser.add_argument(
        "--inpainted",
        help="Path to save inpainted image (optional)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model file (default: new_best.pt in same directory)"
    )

    args = parser.parse_args()

    main(
        image_path=args.image_path,
        output_json=args.output,
        output_inpainted=args.inpainted,
        model_path=args.model
    )
