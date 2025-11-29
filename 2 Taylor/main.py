#!/usr/bin/env python3
"""
Image compression tool that loads bounding boxes from JSON, creates an image
with unique colors for each box, then compresses it by dividing into squares
and replacing each square with its most common color.
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import Counter

from PIL import Image, ImageDraw


@dataclass
class Pixel:
    """Represents a single pixel's color values."""
    r: int
    g: int
    b: int
    a: int = 255

    def __hash__(self):
        return hash((self.r, self.g, self.b, self.a))

    def __eq__(self, other):
        if not isinstance(other, Pixel):
            return False
        return (self.r == other.r and self.g == other.g and
                self.b == other.b and self.a == other.a)


@dataclass
class CompressedImage:
    """Represents the compressed image data."""
    original_width: int
    original_height: int
    square_size: int
    grid_width: int
    grid_height: int
    squares: List[List[Pixel]]


@dataclass
class Symbol:
    """Represents a symbol from the JSON input."""
    id: int
    cls_id: int
    name: str
    conf: float
    bbox: List[float]
    center: List[float]


def load_symbols_from_json(json_path: str) -> List[Symbol]:
    """Load symbols from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    symbols = []
    for symbol_data in data.get('symbols', []):
        symbol = Symbol(
            id=symbol_data['id'],
            cls_id=symbol_data['cls_id'],
            name=symbol_data['name'],
            conf=symbol_data['conf'],
            bbox=symbol_data['bbox'],
            center=symbol_data['center']
        )
        symbols.append(symbol)

    return symbols


def generate_unique_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generate unique colors for each ID.
    Excludes pure white (255, 255, 255) and pure red (255, 0, 0).
    """
    colors = []
    forbidden_colors = {(255, 255, 255), (255, 0, 0)}

    # Use a deterministic seed for reproducibility
    random.seed(42)

    # Generate colors using various strategies to ensure good distribution
    while len(colors) < num_colors:
        # Strategy 1: Evenly distributed hues with high saturation
        if len(colors) < num_colors:
            hue_step = 360 / max(num_colors, 1)
            for i in range(num_colors):
                hue = int(i * hue_step) % 360
                # Convert HSV to RGB (simplified)
                h = hue / 60.0
                c = 255
                x = int(c * (1 - abs(h % 2 - 1)))
                if 0 <= h < 1:
                    r, g, b = c, x, 0
                elif 1 <= h < 2:
                    r, g, b = x, c, 0
                elif 2 <= h < 3:
                    r, g, b = 0, c, x
                elif 3 <= h < 4:
                    r, g, b = 0, x, c
                elif 4 <= h < 5:
                    r, g, b = x, 0, c
                else:
                    r, g, b = c, 0, x

                color = (r, g, b)
                if color not in forbidden_colors and color not in colors:
                    colors.append(color)
                    if len(colors) >= num_colors:
                        break

        # Strategy 2: Add random bright colors if we need more
        while len(colors) < num_colors:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = (r, g, b)

            if color not in forbidden_colors and color not in colors:
                colors.append(color)

    return colors[:num_colors]


def create_image_from_symbols(
    symbols: List[Symbol],
    lines: List[Dict] = None,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    bbox_multiplier: float = 1.0,
    line_width: int = 3
) -> Image.Image:
    """
    Create an image from symbols and lines with each element in a unique color.

    Args:
        symbols: List of Symbol objects
        lines: List of line dictionaries with x1, y1, x2, y2 coordinates (optional)
        background_color: Background color for the image (default: white)
        bbox_multiplier: Scale factor for all bounding boxes (default: 1.0)
                        Values > 1.0 enlarge boxes, values < 1.0 shrink boxes
        line_width: Width of lines to draw in pixels (default: 3)

    Returns:
        PIL Image with bounding boxes and lines drawn
    """
    # Calculate canvas size needed (with multiplier applied)
    max_x = 0
    max_y = 0

    for symbol in symbols:
        bbox = symbol.bbox
        # bbox format: [x1, y1, x2, y2] - apply multiplier
        scaled_bbox = [coord * bbox_multiplier for coord in bbox]
        max_x = max(max_x, scaled_bbox[2])
        max_y = max(max_y, scaled_bbox[3])

    # Also consider line endpoints
    if lines:
        for line in lines:
            max_x = max(max_x, line['x1'] * bbox_multiplier, line['x2'] * bbox_multiplier)
            max_y = max(max_y, line['y1'] * bbox_multiplier, line['y2'] * bbox_multiplier)

    # Add some padding
    width = int(max_x) + 10
    height = int(max_y) + 10

    # Create image with background color
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)

    # Draw lines first (in RED) so they appear below symbols
    # RED: where R > G and R > B (for Ryan's connection finding)
    if lines:
        line_color = (255, 0, 0)  # Pure red for lines
        for line in lines:
            x1 = int(line['x1'] * bbox_multiplier)
            y1 = int(line['y1'] * bbox_multiplier)
            x2 = int(line['x2'] * bbox_multiplier)
            y2 = int(line['y2'] * bbox_multiplier)
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)

    # Draw symbols in GREEN (where G > R and G > B for Ryan's connection finding)
    # All symbols get the SAME pure green color so Ryan's BFS can identify them
    symbol_color = (0, 255, 0)  # Pure green for all symbols

    for symbol in symbols:
        bbox = symbol.bbox

        # Apply multiplier and convert bbox to integer coordinates
        x1 = int(bbox[0] * bbox_multiplier)
        y1 = int(bbox[1] * bbox_multiplier)
        x2 = int(bbox[2] * bbox_multiplier)
        y2 = int(bbox[3] * bbox_multiplier)

        # Fill the bounding box with pure green
        draw.rectangle([x1, y1, x2, y2], fill=symbol_color, outline=symbol_color)

    return img


def image_to_pixel_matrix(img: Image.Image) -> List[List[Pixel]]:
    """Convert an image to a 2D matrix of pixels."""
    width, height = img.size
    pixels = img.convert('RGBA').load()

    matrix = []
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b, a = pixels[x, y]
            row.append(Pixel(r, g, b, a))
        matrix.append(row)

    return matrix


def calculate_rgb_count(matrix: List[List[Pixel]]) -> Tuple[float, float, float]:
    """Calculate the count of pixels with non-zero R, G, B values."""
    red_count = sum(1 for row in matrix for pixel in row if pixel.r > 0)
    blue_count = sum(1 for row in matrix for pixel in row if pixel.b > 0)
    green_count = sum(1 for row in matrix for pixel in row if pixel.g > 0)

    return float(red_count), float(blue_count), float(green_count)


def generate_square_sizes(min_dim: int = 8, max_dim: int = 32) -> List[int]:
    """
    Generate a reasonable set of square sizes for compression.
    Returns sorted array from largest to smallest square size (coarsest to finest).
    """
    sizes = []
    n = min_dim
    while n <= max_dim:
        sizes.append(n)
        n *= 2

    # Sort descending (largest square first = coarsest compression)
    return sorted(set(sizes), reverse=True)


def is_white(pixel: Pixel) -> bool:
    """Check if a pixel is white (all RGB components are 255)."""
    return pixel.r == 255 and pixel.g == 255 and pixel.b == 255


def color_distance(p1: Pixel, p2: Pixel) -> float:
    """Calculate the squared Euclidean distance between two colors."""
    dr = p1.r - p2.r
    dg = p1.g - p2.g
    db = p1.b - p2.b
    return float(dr * dr + dg * dg + db * db)


def get_unique_colors(matrix: List[List[Pixel]]) -> Dict[Pixel, bool]:
    """Return a dictionary of all unique colors in the matrix."""
    return {pixel: True for row in matrix for pixel in row}


def quantize_colors(matrix: List[List[Pixel]], threshold: float) -> List[List[Pixel]]:
    """
    Reduce color variations by snapping similar colors together.

    Args:
        matrix: 2D pixel matrix
        threshold: Controls how aggressive the quantization is (lower = more aggressive)
                  Common values: 100-500 for light cleanup, 500-2000 for moderate, 2000+ for aggressive

    Returns:
        Quantized pixel matrix
    """
    height = len(matrix)
    width = len(matrix[0])

    # Collect all unique colors and their frequencies
    color_freq = Counter(pixel for row in matrix for pixel in row)

    # Sort colors by frequency (most common first)
    colors_sorted = sorted(color_freq.items(), key=lambda x: x[1], reverse=True)

    # Create color mapping: similar colors map to their dominant representative
    color_map = {}

    for current_color, _ in colors_sorted:
        # If this color already has a mapping, skip it
        if current_color in color_map:
            continue

        # This color maps to itself
        color_map[current_color] = current_color

        # Find all similar colors and map them to this dominant color
        for other_color, _ in colors_sorted:
            # Skip if already mapped
            if other_color in color_map:
                continue

            # If colors are similar enough, map to the dominant color
            if color_distance(current_color, other_color) <= threshold:
                color_map[other_color] = current_color

    # Apply the color mapping
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            original_color = matrix[y][x]
            mapped_color = color_map.get(original_color, original_color)
            row.append(mapped_color)
        result.append(row)

    return result


def find_most_common_color(
    matrix: List[List[Pixel]],
    start_y: int,
    start_x: int,
    square_size: int
) -> Pixel:
    """
    Find the most common color in a square region.
    White pixels are ignored unless all pixels are white.
    GREEN pixels (symbols) have priority over RED pixels (lines).
    Handles partial squares at image boundaries.
    """
    height = len(matrix)
    width = len(matrix[0])

    # Calculate actual bounds (may be smaller than square_size at edges)
    end_y = min(start_y + square_size, height)
    end_x = min(start_x + square_size, width)

    # Count occurrences of each color
    color_count = Counter()
    has_non_white = False
    has_green = False

    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            pixel = matrix[y][x]
            if not is_white(pixel):
                has_non_white = True
                # Check if this is green (G > R and G > B)
                if pixel.g > pixel.r and pixel.g > pixel.b:
                    has_green = True
            color_count[pixel] += 1

    # Priority 1: If there's any green (symbols), return pure green
    # This ensures symbols are preserved over lines during compression
    if has_green:
        return Pixel(0, 255, 0, 255)

    # Priority 2: If there are non-white pixels but no green, find most common
    if has_non_white:
        non_white_colors = [(pixel, count) for pixel, count in color_count.items()
                           if not is_white(pixel)]
        most_common = max(non_white_colors, key=lambda x: x[1])[0]
        return most_common

    # All pixels are white
    return Pixel(255, 255, 255, 255)


def slice_and_compress(matrix: List[List[Pixel]], square_size: int) -> CompressedImage:
    """
    Compress the image by dividing it into nxn squares and replacing each square
    with its most common color. Allows partial squares at edges for non-evenly-divisible dimensions.
    """
    height = len(matrix)
    width = len(matrix[0])

    # Calculate grid dimensions using ceiling division to include partial squares
    grid_height = (height + square_size - 1) // square_size
    grid_width = (width + square_size - 1) // square_size

    # Create compressed squares array
    compressed = []

    # Process each square (including partial ones at edges)
    for grid_y in range(grid_height):
        row = []
        for grid_x in range(grid_width):
            start_y = grid_y * square_size
            start_x = grid_x * square_size
            pixel = find_most_common_color(matrix, start_y, start_x, square_size)
            row.append(pixel)
        compressed.append(row)

    return CompressedImage(
        original_width=width,
        original_height=height,
        square_size=square_size,
        grid_width=grid_width,
        grid_height=grid_height,
        squares=compressed
    )


def compressed_to_image(compressed: CompressedImage) -> Image.Image:
    """
    Convert a CompressedImage back to a drawable image.
    Each square is expanded to its original size with the dominant color.
    Handles partial squares at image boundaries.
    """
    img = Image.new('RGBA', (compressed.original_width, compressed.original_height))
    pixels = img.load()

    # Fill each square with its color
    for grid_y in range(compressed.grid_height):
        for grid_x in range(compressed.grid_width):
            pixel = compressed.squares[grid_y][grid_x]

            # Fill the square region (handle partial squares at edges)
            start_y = grid_y * compressed.square_size
            start_x = grid_x * compressed.square_size
            end_y = min(start_y + compressed.square_size, compressed.original_height)
            end_x = min(start_x + compressed.square_size, compressed.original_width)

            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    pixels[x, y] = (pixel.r, pixel.g, pixel.b, pixel.a)

    return img


def save_compressed_images(
    compressions: List[CompressedImage],
    valid_sizes: List[int],
    output_dir: Path = None
) -> None:
    """Save all compressed images to a timestamped output folder."""
    # Create timestamped folder if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("output") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving compressed images to: {output_dir}")

    # Save each compression level
    for i, comp in enumerate(compressions):
        # Convert to image
        img = compressed_to_image(comp)

        # Create filename based on square size
        filename = output_dir / f"compressed_{valid_sizes[i]}x{valid_sizes[i]}.png"

        # Save as PNG
        img.save(filename, 'PNG')

        print(f"  Saved: compressed_{valid_sizes[i]}x{valid_sizes[i]}.png "
              f"({comp.grid_width}x{comp.grid_height} grid)")


def create_output_json(
    symbols: List[Symbol],
    compressions: List[CompressedImage],
    compression_sizes: List[int],
    original_width: int,
    original_height: int,
    bbox_multiplier: float,
    enable_color_quantization: bool,
    color_threshold: float,
    original_colors: int,
    quantized_colors: int
) -> Dict[str, Any]:
    """
    Create a JSON-serializable dictionary with compression results.

    Args:
        symbols: List of symbols from input
        compressions: List of compressed images
        compression_sizes: List of square sizes used
        original_width: Width of generated image
        original_height: Height of generated image
        bbox_multiplier: Bounding box scale factor used
        enable_color_quantization: Whether color quantization was enabled
        color_threshold: Threshold used for color quantization
        original_colors: Number of colors before quantization
        quantized_colors: Number of colors after quantization

    Returns:
        Dictionary containing all compression data
    """
    output_data = {
        "settings": {
            "bbox_multiplier": bbox_multiplier,
            "color_quantization_enabled": enable_color_quantization,
            "color_threshold": color_threshold,
            "compression_sizes": compression_sizes
        },
        "input_data": {
            "num_symbols": len(symbols),
            "symbol_ids": [s.id for s in symbols],
            "symbol_names": [s.name for s in symbols]
        },
        "image_info": {
            "width": original_width,
            "height": original_height,
            "total_pixels": original_width * original_height,
            "original_colors": original_colors,
            "quantized_colors": quantized_colors if enable_color_quantization else original_colors,
            "color_reduction_percent": round((original_colors - quantized_colors) * 100 / original_colors, 2) if original_colors > 0 else 0
        },
        "compressions": []
    }

    # Add compression data for each level
    for i, comp in enumerate(compressions):
        compression_percent = round((comp.grid_width * comp.grid_height) * 100 / (original_width * original_height), 4)

        # Convert compressed pixel data to serializable format
        compressed_pixels = []
        for row in comp.squares:
            pixel_row = []
            for pixel in row:
                pixel_row.append({
                    "r": pixel.r,
                    "g": pixel.g,
                    "b": pixel.b,
                    "a": pixel.a
                })
            compressed_pixels.append(pixel_row)

        compression_data = {
            "index": i,
            "square_size": compression_sizes[i],
            "grid_width": comp.grid_width,
            "grid_height": comp.grid_height,
            "total_squares": comp.grid_width * comp.grid_height,
            "compression_percent": compression_percent,
            "compressed_pixels": compressed_pixels
        }
        output_data["compressions"].append(compression_data)

    return output_data


def save_json_output(output_data: Dict[str, Any], output_path: Path) -> None:
    """
    Save output data to a JSON file.

    Args:
        output_data: Dictionary containing all output data
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved JSON output to: {output_path.name}")


def main(json_input_file=None, lines_json_file=None, output_dir=None, bbox_multiplier=1.0):
    """Main entry point for the image compression tool.

    Args:
        json_input_file: Path to input JSON file with symbols (default: bs_connected.json)
        lines_json_file: Path to input JSON file with lines (optional)
        output_dir: Output directory path (default: timestamped output/ folder)
        bbox_multiplier: Scale factor for bounding boxes (default: 1.0)
    """
    # Configuration
    if json_input_file is None:
        json_input_file = "bs_connected.json"

    debug_output_images = True

    # Preprocessing flag - set to True to clean up color variations/anti-aliasing
    enable_color_quantization = True
    color_threshold = 500.0  # Lower = more aggressive cleanup
                             # (100-500: light, 500-2000: moderate, 2000+: aggressive)

    # Load symbols from JSON
    try:
        symbols = load_symbols_from_json(json_input_file)
        print(f"Loaded {len(symbols)} symbols from {json_input_file}")
    except FileNotFoundError:
        print(f"Error: {json_input_file} not found")
        return
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Load lines from JSON if provided
    lines = []
    if lines_json_file:
        try:
            with open(lines_json_file, 'r') as f:
                lines_data = json.load(f)
                lines = lines_data.get('lines', [])
            print(f"Loaded {len(lines)} lines from {lines_json_file}")
        except FileNotFoundError:
            print(f"Warning: {lines_json_file} not found, continuing without lines")
        except Exception as e:
            print(f"Warning: Error loading lines JSON: {e}, continuing without lines")

    # Create image from symbols and lines
    print(f"\nCreating image from {len(symbols)} bounding boxes and {len(lines)} lines (multiplier: {bbox_multiplier})...")
    img = create_image_from_symbols(symbols, lines=lines, bbox_multiplier=bbox_multiplier)

    # Get image dimensions
    width, height = img.size
    print(f"Generated image size: {width}x{height}")

    # Save the original generated image
    if debug_output_images:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = Path("output") / timestamp
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        original_img_path = output_dir / "original_bboxes.png"
        img.save(original_img_path)
        print(f"Saved original bounding box image to: {original_img_path}")

    # Convert to pixel matrix
    matrix = image_to_pixel_matrix(img)

    # Track color counts
    original_colors = len(get_unique_colors(matrix))
    quantized_colors = original_colors

    # Optional: Quantize colors to remove shading/anti-aliasing
    if enable_color_quantization:
        print(f"\nApplying color quantization (threshold: {color_threshold:.0f})...")
        print(f"Original colors: {original_colors}")
        matrix = quantize_colors(matrix, color_threshold)
        quantized_colors = len(get_unique_colors(matrix))
        reduction_pct = (original_colors - quantized_colors) * 100 / original_colors
        print(f"Reduced colors from {original_colors} to {quantized_colors} "
              f"({reduction_pct:.1f}% reduction)")

    # Calculate total of each color value
    red_count, blue_count, green_count = calculate_rgb_count(matrix)
    print(f"Red count is: {red_count:.2f}")
    print(f"Blue count is: {blue_count:.2f}")
    print(f"Green count is: {green_count:.2f}")

    # Generate reasonable compression sizes
    compression_sizes = generate_square_sizes()
    print(f"\nGenerated {len(compression_sizes)} compression sizes: {compression_sizes}")

    # Generate compressions at all sizes
    print("\nGenerating compressions at all sizes...")
    compressions = []

    for i, size in enumerate(compression_sizes):
        comp = slice_and_compress(matrix, size)
        compressions.append(comp)
        compression_pct = (comp.grid_width * comp.grid_height) * 100 / (width * height)
        print(f"  [{i}] {size}x{size} squares -> {comp.grid_width}x{comp.grid_height} grid "
              f"(compression: {compression_pct:.2f}%)")

    print("\nAll compressions generated! You can access them via the compressions list.")
    print("compressions[0] = smallest grid (largest squares)")
    print(f"compressions[{len(compressions) - 1}] = largest grid (smallest squares)")

    # Optionally save debug images
    if debug_output_images:
        try:
            save_compressed_images(compressions, compression_sizes, output_dir)
            print("\nAll images saved successfully!")
        except Exception as e:
            print(f"Error saving debug images: {e}")

    # Create and save JSON output
    print("\nGenerating JSON output...")
    output_data = create_output_json(
        symbols=symbols,
        compressions=compressions,
        compression_sizes=compression_sizes,
        original_width=width,
        original_height=height,
        bbox_multiplier=bbox_multiplier,
        enable_color_quantization=enable_color_quantization,
        color_threshold=color_threshold,
        original_colors=original_colors,
        quantized_colors=quantized_colors
    )

    json_output_path = output_dir / "compression_results.json"
    save_json_output(output_data, json_output_path)
    print(f"\nProcessing complete! All outputs saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Image compression tool for pneumatic/hydraulic diagrams"
    )

    parser.add_argument(
        '--input-json',
        default="bs_connected.json",
        help='Path to input JSON file with symbol definitions (default: bs_connected.json)'
    )

    parser.add_argument(
        '--lines-json',
        help='Path to input JSON file with line definitions (optional)'
    )

    parser.add_argument(
        '--output-dir',
        help='Output directory path (default: timestamped output/ folder)'
    )

    parser.add_argument(
        '--bbox-multiplier',
        type=float,
        default=1.0,
        help='Scale factor for bounding boxes (default: 1.0)'
    )

    args = parser.parse_args()

    main(
        json_input_file=args.input_json,
        lines_json_file=args.lines_json,
        output_dir=args.output_dir,
        bbox_multiplier=args.bbox_multiplier
    )
