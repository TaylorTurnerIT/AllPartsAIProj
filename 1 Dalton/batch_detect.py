#!/usr/bin/env python3
"""
Batch detection wrapper for the pipeline.
Runs symbol detection and outputs JSON in the expected format.

Usage:
    python batch_detect.py INPUT_IMAGE --output OUTPUT_JSON
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2


class SymbolDetector:
    """Wrapper for YOLO-based symbol detection."""

    def __init__(self, model_path, conf_threshold=0.25):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.model = None

        # Detection parameters (from run_new_best.py)
        self.IMG_SIZE = 1600
        self.IOU = 0.45
        self.MERGE_IOU_BREAKER = 0.4
        self.MERGE_IOU_TRANSFORMER = 0.45
        self.BOX_SHRINK_BREAKER = 0.85
        self.BOX_SHRINK_TRANSFORMER = 0.8
        self.TRANSFORMER_CONF_MIN = 0.3
        self.BREAKER_CONF_MIN = 0.25
        self.TRANSFORMER_Y_MAX_FRAC = 0.7
        self.KEEP_TOP_TRANSFORMERS = 2
        self.KEEP_TOP_BREAKERS = 24
        self.INPAINT_PAD = 4
        self.INPAINT_RADIUS = 3

    def load_model(self):
        """Load the YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def iou(self, box_a, box_b):
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

    def merge_boxes(self, boxes):
        """Greedy merge: keep highest-conf per class, drop overlapping boxes."""
        boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
        kept = []
        for b in boxes:
            cls = b["name"]
            thr = self.MERGE_IOU_BREAKER if cls == "breaker" else self.MERGE_IOU_TRANSFORMER
            if any(b["cls_id"] == k["cls_id"] and self.iou(b["bbox"], k["bbox"]) >= thr for k in kept):
                continue
            kept.append(b)
        return kept

    def shrink_box(self, bbox, shrink, width, height):
        """Shrink a box around its center."""
        if shrink >= 1.0:
            return bbox
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1) * shrink
        h = (y2 - y1) * shrink
        nx1 = max(0.0, cx - w / 2)
        ny1 = max(0.0, cy - h / 2)
        nx2 = min(width, cx + w / 2)
        ny2 = min(height, cy + h / 2)
        return [nx1, ny1, nx2, ny2]

    def filter_boxes(self, boxes, width, height):
        """Filter out obviously bad boxes."""
        filtered = []
        for b in boxes:
            if b["name"] == "transformer":
                _, y1, _, y2 = b["bbox"]
                y_center = ((y1 + y2) / 2) / height
                if y_center > self.TRANSFORMER_Y_MAX_FRAC:
                    continue
            filtered.append(b)
        return filtered

    def detect(self, image_path, output_json_path, output_inpainted_path=None):
        """
        Run detection on an image and save results.

        Args:
            image_path: Path to input image
            output_json_path: Path to save JSON output
            output_inpainted_path: Path to save inpainted image (symbols removed)
        """
        if self.model is None:
            self.load_model()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Load image
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        # Run detection on original orientation
        result = self.model.predict(
            source=str(image_path),
            imgsz=self.IMG_SIZE,
            conf=self.conf_threshold,
            iou=self.IOU,
            save=False,
            device="cpu",
            verbose=False
        )[0]

        # Extract boxes
        boxes = []
        for box in result.boxes:
            cls_id = int(box.cls)
            cls_name = result.names[cls_id]
            conf = float(box.conf)
            xyxy = [float(x) for x in box.xyxy[0]]

            # Apply confidence thresholds
            if cls_name == "transformer" and conf < self.TRANSFORMER_CONF_MIN:
                continue
            if cls_name == "breaker" and conf < self.BREAKER_CONF_MIN:
                continue

            boxes.append({
                "cls_id": cls_id,
                "name": cls_name,
                "conf": conf,
                "bbox": xyxy
            })

        # Merge overlapping boxes
        boxes = self.merge_boxes(boxes)
        boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
        boxes = self.filter_boxes(boxes, width, height)

        # Keep top-K per class
        keep = []
        breaker_count = 0
        transformer_count = 0
        for b in boxes:
            if b["name"] == "breaker":
                if breaker_count >= self.KEEP_TOP_BREAKERS:
                    continue
                breaker_count += 1
            elif b["name"] == "transformer":
                if transformer_count >= self.KEEP_TOP_TRANSFORMERS:
                    continue
                transformer_count += 1
            keep.append(b)
        boxes = keep

        # Shrink boxes and calculate centers
        symbols = []
        for idx, b in enumerate(boxes):
            shrink = self.BOX_SHRINK_BREAKER if b["name"] == "breaker" else self.BOX_SHRINK_TRANSFORMER
            shrunk_bbox = self.shrink_box(b["bbox"], shrink, width, height)

            # Calculate center
            center = [
                (shrunk_bbox[0] + shrunk_bbox[2]) / 2,
                (shrunk_bbox[1] + shrunk_bbox[3]) / 2
            ]

            symbols.append({
                "id": idx,
                "cls_id": b["cls_id"],
                "name": b["name"],
                "conf": b["conf"],
                "bbox": shrunk_bbox,
                "center": center
            })

        # Create output JSON
        output_data = {
            "symbols": symbols,
            "lines": [],  # Will be filled by line detection
            "connections": []  # Will be filled by connection finding
        }

        # Save JSON
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Detected {len(symbols)} symbols")
        print(f"Saved to: {output_json_path}")

        # Create inpainted image (symbols removed for line detection)
        if output_inpainted_path:
            self.create_inpainted_image(image_path, symbols, output_inpainted_path)

        return output_data

    def create_inpainted_image(self, image_path, symbols, output_path):
        """Remove detected symbols via inpainting to leave lines for tracing."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Create mask for inpainting
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for symbol in symbols:
            x1, y1, x2, y2 = symbol["bbox"]
            x1 = max(0, int(round(x1 - self.INPAINT_PAD)))
            y1 = max(0, int(round(y1 - self.INPAINT_PAD)))
            x2 = min(img.shape[1] - 1, int(round(x2 + self.INPAINT_PAD)))
            y2 = min(img.shape[0] - 1, int(round(y2 + self.INPAINT_PAD)))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

        # Inpaint
        inpainted = cv2.inpaint(img, mask, self.INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(str(output_path), inpainted)
        print(f"Saved inpainted image: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch symbol detection for pipeline")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--inpainted", help="Path to save inpainted image (optional)")
    parser.add_argument("--model", default=None, help="Path to YOLO model (default: new_best.pt in same dir)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path(__file__).parent / "new_best.pt"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Run detection
    detector = SymbolDetector(model_path, conf_threshold=args.conf)
    detector.detect(args.input_image, args.output, args.inpainted)


if __name__ == "__main__":
    main()
