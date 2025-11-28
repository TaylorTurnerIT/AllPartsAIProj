"""
Connect detected symbols to traced lines and render an overlay.

Inputs:
 - Original image: bs.png
 - Symbol labels: runs/detect/new_best_manual_merged/labels/bs.txt (YOLO txt with conf)
 - Lines: runs/lines/bs_lines.json (from line_trace.py)

Outputs (under runs/lines):
 - bs_connected.json : symbols, lines, and nearest-line connection per symbol
 - bs_connected.jpg  : overlay with lines (red), symbols (blue boxes), and connectors (cyan)

Usage:
    ./py312/bin/python connect_symbols.py
"""

from pathlib import Path
import json
import math

import cv2
import numpy as np

IMG_PATH = Path("bs.png")
LABELS_PATH = Path("runs/detect/new_best_manual_merged/labels/bs.txt")
LINES_PATH = Path("runs/lines/bs_lines.json")
OUT_DIR = Path("runs/lines")
CLASS_MAP = {0: "transformer", 1: "breaker"}
CONNECT_DIST = 80.0  # max pixel distance from symbol center to a line to consider connected
BOX_COLOR = (0, 128, 255)
LINE_COLOR = (0, 0, 255)
CONNECT_COLOR = (0, 255, 255)


def load_symbols(img_w: int, img_h: int):
    symbols = []
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")
    for idx, line in enumerate(LABELS_PATH.read_text().splitlines()):
        parts = line.split()
        if len(parts) < 6:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h, conf = map(float, parts[1:6])
        name = CLASS_MAP.get(cls_id, str(cls_id))
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        cx = xc * img_w
        cy = yc * img_h
        symbols.append(
            {
                "id": idx,
                "cls_id": cls_id,
                "name": name,
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
            }
        )
    return symbols


def load_lines():
    if not LINES_PATH.exists():
        raise FileNotFoundError(f"Lines not found: {LINES_PATH}")
    data = json.loads(LINES_PATH.read_text())
    return data.get("lines", [])


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Return distance and closest point from p to segment (x1,y1)-(x2,y2)."""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0:
        return math.hypot(px - x1, py - y1), (x1, y1)
    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    dist = math.hypot(px - proj_x, py - proj_y)
    return dist, (proj_x, proj_y)


def connect(symbols, lines):
    connections = []
    for s in symbols:
        cx, cy = s["center"]
        best = None
        for idx, ln in enumerate(lines):
            x1, y1, x2, y2 = ln["x1"], ln["y1"], ln["x2"], ln["y2"]
            dist, (ax, ay) = point_to_segment_distance(cx, cy, x1, y1, x2, y2)
            if best is None or dist < best["distance"]:
                best = {"line_id": idx, "distance": dist, "attach": [ax, ay]}
        if best and best["distance"] <= CONNECT_DIST:
            connections.append({"symbol_id": s["id"], **best})
    return connections


def draw_overlay(img, symbols, lines, connections, out_path: Path):
    canvas = img.copy()
    # Draw lines
    for ln in lines:
        cv2.line(canvas, (int(ln["x1"]), int(ln["y1"])), (int(ln["x2"]), int(ln["y2"])), LINE_COLOR, 2)
    # Draw symbols
    for s in symbols:
        x1, y1, x2, y2 = map(int, s["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cx, cy = map(int, s["center"])
        cv2.circle(canvas, (cx, cy), 3, BOX_COLOR, -1)
        label = f"{s['name']} {s['conf']:.2f}"
        cv2.putText(canvas, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1, cv2.LINE_AA)
    # Draw connections
    for c in connections:
        s = symbols[c["symbol_id"]]
        cx, cy = map(int, s["center"])
        ax, ay = map(int, c["attach"])
        cv2.line(canvas, (cx, cy), (ax, ay), CONNECT_COLOR, 1, cv2.LINE_AA)
        cv2.circle(canvas, (ax, ay), 3, CONNECT_COLOR, -1)
    cv2.imwrite(str(out_path), canvas)


def main():
    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")
    img = cv2.imread(str(IMG_PATH), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read {IMG_PATH}")
    h, w = img.shape[:2]

    symbols = load_symbols(w, h)
    lines = load_lines()
    connections = connect(symbols, lines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "bs_connected.json"
    out_img = OUT_DIR / "bs_connected.jpg"

    draw_overlay(img, symbols, lines, connections, out_img)

    payload = {"symbols": symbols, "lines": lines, "connections": connections}
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"Overlay: {out_img}")
    print(f"JSON: {out_json}")
    print(f"Symbols: {len(symbols)}, Lines: {len(lines)}, Connections: {len(connections)}")


if __name__ == "__main__":
    main()
