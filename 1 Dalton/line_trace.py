"""
Quick line tracer for cleaned schematic images.

Input: runs/detect/new_best_manual_merged/bs_inpainted.png
Outputs (under runs/lines):
 - bs_edges.png: Canny edges on inverted mask
 - bs_skel.png: thinned skeleton
 - bs_lines_overlay.png: Hough lines drawn on original gray
 - bs_lines.json: line endpoints with length

Usage:
    ./py312/bin/python line_trace.py
"""

from pathlib import Path
import json
import math

import cv2
import numpy as np

IMG_PATH = Path("runs/detect/new_best_manual_merged/bs_inpainted.png")
OUT_DIR = Path("runs/lines")

# Hough parameters (tune if needed)
HOUGH_THRESHOLD = 60
HOUGH_MIN_LINE_LENGTH = 40
HOUGH_MAX_LINE_GAP = 8
GAUSS_BLUR = 3  # must be odd; 0 disables


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Return morphological skeleton of a binary mask (1-channel, values 0/255)."""
    size = np.size(binary)
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    img = binary.copy()
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


def main():
    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMG_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gray = cv2.imread(str(IMG_PATH), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read {IMG_PATH}")

    # Binarize and invert so lines are white on black
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw
    if GAUSS_BLUR and GAUSS_BLUR > 0:
        inv = cv2.GaussianBlur(inv, (GAUSS_BLUR, GAUSS_BLUR), 0)

    edges = cv2.Canny(inv, 50, 150, apertureSize=3)
    cv2.imwrite(str(OUT_DIR / "bs_edges.png"), edges)

    skel = skeletonize(inv)
    cv2.imwrite(str(OUT_DIR / "bs_skel.png"), skel)

    # Hough lines on edges for robustness
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out_lines = []
    if lines is not None:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            length = math.hypot(x2 - x1, y2 - y1)
            out_lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": length})

    cv2.imwrite(str(OUT_DIR / "bs_lines_overlay.png"), overlay)
    (OUT_DIR / "bs_lines.json").write_text(json.dumps({"lines": out_lines}, indent=2))

    print(f"Input: {IMG_PATH}")
    print(f"Edges: {OUT_DIR / 'bs_edges.png'}")
    print(f"Skeleton: {OUT_DIR / 'bs_skel.png'}")
    print(f"Overlay: {OUT_DIR / 'bs_lines_overlay.png'}")
    print(f"JSON: {OUT_DIR / 'bs_lines.json'} ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
