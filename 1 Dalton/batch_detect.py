#!/usr/bin/env python3
"""
Batch detection wrapper that uses run_new_best.py logic.
Converts the output to the JSON format expected by the pipeline.

Usage:
    python batch_detect.py INPUT_IMAGE --output OUTPUT_JSON [--inpainted INPAINTED_IMAGE]
"""

import argparse
import json
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# Import detection functions from run_new_best.py
def map_box_back(angle, xyxy, width, height):
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


def iou(box_a, box_b):
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


def merge_boxes(boxes, iou_thresh_breaker=0.4, iou_thresh_transformer=0.45):
    """Greedy merge: keep highest-conf per class, drop boxes that overlap too much."""
    boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    kept = []
    for b in boxes:
        cls = b["name"]
        thr = iou_thresh_breaker if cls == "breaker" else iou_thresh_transformer
        if any(b["cls"] == k["cls"] and iou(b["xyxy"], k["xyxy"]) >= thr for k in kept):
            continue
        kept.append(b)
    return kept


def shrink_box(xyxy, shrink, width, height):
    """Shrink a box around its center by a factor."""
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


def filter_boxes(boxes, width, height, transformer_y_max_frac=0.7):
    """Prune obviously bad boxes with simple heuristics."""
    filtered = []
    for b in boxes:
        if b["name"] == "transformer":
            _, y1, _, y2 = b["xyxy"]
            y_center = ((y1 + y2) / 2) / height
            if y_center > transformer_y_max_frac:
                continue
        filtered.append(b)
    return filtered


def inpaint_symbols(image_path, boxes, output_path, inpaint_pad=4, inpaint_radius=3):
    """Remove detected symbols via inpainting to leave lines for tracing."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        x1 = max(0, int(round(x1 - inpaint_pad)))
        y1 = max(0, int(round(y1 - inpaint_pad)))
        x2 = min(img.shape[1] - 1, int(round(x2 + inpaint_pad)))
        y2 = min(img.shape[0] - 1, int(round(y2 + inpaint_pad)))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    inpainted = cv2.inpaint(img, mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(str(output_path), inpainted)
    return output_path


def run_detection(image_path, model_path):
    """
    Run detection using the same logic as run_new_best.py.

    Returns:
        List of detected boxes with format:
        [{"cls": int, "name": str, "conf": float, "xyxy": [x1, y1, x2, y2]}]
    """
    # Parameters from run_new_best.py
    IMG_SIZE = 1600
    CONF = 0.25
    IOU = 0.45
    MERGE_IOU_BREAKER = 0.4
    MERGE_IOU_TRANSFORMER = 0.45
    ROTATED_CLASSES = {"breaker"}
    ROTATED_SUPPRESS_IOU = 0.5
    ROTATED_CONF_MIN = 0.4
    ROTATED_TOPK = 30
    TRANSFORMER_CONF_MIN = 0.3
    BREAKER_CONF_MIN = 0.25
    KEEP_TOP_TRANSFORMERS = 2
    KEEP_TOP_BREAKERS = 24
    BOX_SHRINK_BREAKER = 0.85
    BOX_SHRINK_TRANSFORMER = 0.8
    TRANSFORMER_Y_MAX_FRAC = 0.7

    # Load image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Load model
    model = YOLO(str(model_path))

    # Run detection on multiple orientations
    orientations = [
        ("main", 0),
        ("rot90", 90),
        ("rot270", 270),
    ]

    main_boxes = []
    rotated_boxes = []

    for tag, angle in orientations:
        if angle == 0:
            img_src = str(image_path)
        else:
            img_src = np.array(img.rotate(angle, expand=True))

        res = model.predict(
            source=img_src,
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            save=False,
            device="cpu",
            verbose=False,
        )[0]

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

    # Merge boxes
    merged = merge_boxes(
        main_boxes + rotated_boxes,
        iou_thresh_breaker=MERGE_IOU_BREAKER,
        iou_thresh_transformer=MERGE_IOU_TRANSFORMER,
    )
    merged = sorted(merged, key=lambda b: b["conf"], reverse=True)
    merged = filter_boxes(merged, width, height, TRANSFORMER_Y_MAX_FRAC)

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

    # Shrink boxes
    shrunk = []
    for b in merged:
        shrink = BOX_SHRINK_BREAKER if b["name"] == "breaker" else BOX_SHRINK_TRANSFORMER
        shrunk_xyxy = shrink_box(b["xyxy"], shrink, width, height)
        shrunk.append({**b, "xyxy": shrunk_xyxy})

    return shrunk


def main():
    parser = argparse.ArgumentParser(description="Batch symbol detection using run_new_best.py logic")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--inpainted", help="Path to save inpainted image (optional)")
    parser.add_argument("--model", default=None, help="Path to YOLO model (default: new_best.pt in same dir)")

    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path(__file__).parent / "new_best.pt"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Set YOLO config dir to avoid clutter
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))

    try:
        # Run detection
        print(f"Running detection on {args.input_image}...")
        boxes = run_detection(args.input_image, model_path)
        print(f"Detected {len(boxes)} symbols")

        # Convert to JSON format
        symbols = []
        for idx, b in enumerate(boxes):
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

        # Create output JSON
        output_data = {
            "symbols": symbols,
            "lines": [],  # Will be filled by line detection
            "connections": []  # Will be filled by connection finding
        }

        # Save JSON
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to: {output_path}")

        # Create inpainted image if requested
        if args.inpainted:
            print(f"Creating inpainted image...")
            inpaint_symbols(args.input_image, boxes, args.inpainted)
            print(f"Saved inpainted image: {args.inpainted}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
