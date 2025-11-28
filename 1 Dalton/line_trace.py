"""
Quick line tracer for cleaned schematic images.

Detects lines in an inpainted image (with symbols removed) using Hough transform.

Outputs:
 - edges.png: Canny edges on inverted mask
 - skeleton.png: thinned skeleton
 - lines_overlay.png: Hough lines drawn on original gray
 - lines.json: line endpoints with length

Usage:
    python line_trace.py INPUT_IMAGE --output OUTPUT_JSON [--debug-dir DEBUG_DIR]
"""

from pathlib import Path
import json
import math
import argparse
import sys

import cv2
import numpy as np

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


def detect_lines(image_path, output_json_path, debug_dir=None):
    """
    Detect lines in an image using Hough transform.

    Args:
        image_path: Path to input image (should be inpainted with symbols removed)
        output_json_path: Path to save JSON output
        debug_dir: Optional directory to save debug images
    """
    image_path = Path(image_path)
    output_json_path = Path(output_json_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read {image_path}")

    # Binarize and invert so lines are white on black
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw
    if GAUSS_BLUR and GAUSS_BLUR > 0:
        inv = cv2.GaussianBlur(inv, (GAUSS_BLUR, GAUSS_BLUR), 0)

    edges = cv2.Canny(inv, 50, 150, apertureSize=3)
    skel = skeletonize(inv)

    # Hough lines on edges for robustness
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    out_lines = []
    if lines is not None:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            length = math.hypot(x2 - x1, y2 - y1)
            out_lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": length})

    # Save JSON output
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump({"lines": out_lines}, f, indent=2)

    print(f"Detected {len(out_lines)} lines")
    print(f"Saved to: {output_json_path}")

    # Optionally save debug images
    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(debug_dir / "edges.png"), edges)
        cv2.imwrite(str(debug_dir / "skeleton.png"), skel)

        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for l in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, l)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "lines_overlay.png"), overlay)

        print(f"Debug images saved to: {debug_dir}")

    return out_lines


def main():
    parser = argparse.ArgumentParser(description="Detect lines in schematic images")
    parser.add_argument("input_image", help="Path to input image (inpainted)")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--debug-dir", help="Directory to save debug images (optional)")

    args = parser.parse_args()

    try:
        detect_lines(args.input_image, args.output, args.debug_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
