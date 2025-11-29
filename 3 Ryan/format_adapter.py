#!/usr/bin/env python3
"""
Format Adapter: Taylor's Compression Output → Ryan's Connection Finder Input.

Converts between the two JSON formats:
- Taylor outputs: snake_case with lowercase field names
- Ryan expects: PascalCase with uppercase field names

Usage:
    python format_adapter.py INPUT_JSON OUTPUT_JSON
"""

import argparse
import json
import sys
from pathlib import Path


def convert_taylor_to_ryan(taylor_data, compression_index=0):
    """
    Convert Taylor's compression format to Ryan's expected format.

    Args:
        taylor_data: Dictionary from Taylor's compression_results.json
        compression_index: Which compression level to use (default: 0 for first/smallest)

    Returns:
        Dictionary in Ryan's expected format

    Taylor's format:
    {
        "compressions": [{
            "grid_width": 50,
            "grid_height": 40,
            "square_size": 16,
            "compressed_pixels": [[{"r": 255, "g": 0, "b": 0}]]
        }]
    }

    Ryan's format:
    {
        "GridWidth": 50,
        "GridHeight": 40,
        "SquareSize": 16,
        "Squares": [[{"R": 255, "G": 0, "B": 0}]]
    }
    """
    if "compressions" not in taylor_data:
        raise ValueError("Input JSON missing 'compressions' field")

    compressions = taylor_data["compressions"]
    if not compressions:
        raise ValueError("No compression levels found in input JSON")

    if compression_index >= len(compressions):
        raise ValueError(
            f"Compression index {compression_index} out of range "
            f"(available: 0-{len(compressions)-1})"
        )

    compression = compressions[compression_index]

    # Convert field names and structure
    ryan_data = {
        "GridWidth": compression["grid_width"],
        "GridHeight": compression["grid_height"],
        "SquareSize": compression["square_size"],
        "Squares": []
    }

    # Convert pixel grid
    for row in compression["compressed_pixels"]:
        ryan_row = []
        for pixel in row:
            ryan_pixel = {
                "R": pixel["r"],
                "G": pixel["g"],
                "B": pixel["b"]
            }
            ryan_row.append(ryan_pixel)
        ryan_data["Squares"].append(ryan_row)

    return ryan_data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Taylor's compression format to Ryan's expected format"
    )

    parser.add_argument(
        'input_json',
        help='Path to Taylor\'s compression_results.json'
    )

    parser.add_argument(
        'output_json',
        help='Path to output JSON file for Ryan\'s connection finder'
    )

    parser.add_argument(
        '--compression-index',
        type=int,
        default=0,
        help='Which compression level to use (default: 0 for smallest grid)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    try:
        # Read Taylor's output
        with open(args.input_json) as f:
            taylor_data = json.load(f)

        if args.verbose:
            print(f"Loaded Taylor's format from: {args.input_json}")
            print(f"Using compression index: {args.compression_index}")

        # Convert format
        ryan_data = convert_taylor_to_ryan(taylor_data, args.compression_index)

        if args.verbose:
            print(f"Grid size: {ryan_data['GridWidth']}x{ryan_data['GridHeight']}")
            print(f"Square size: {ryan_data['SquareSize']}")

        # Write Ryan's format
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(ryan_data, f, indent=2)

        print(f"Converted format: {output_path}")
        sys.exit(0)

    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_json}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
