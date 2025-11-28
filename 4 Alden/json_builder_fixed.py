#!/usr/bin/env python3
"""
Fixed JSON Builder for Diagram Construction.

This module enhances Ryan's graph.json output with additional symbol metadata
from the AI recognition (bs_connected.json) and compression data.

Key fixes from original json_builder.py:
1. ✓ Loads matrix from compression_results.json (fixes undefined matrix issue)
2. ✓ Proper Path handling instead of string.exists()
3. ✓ Integrates with Ryan's graph.json for connections
4. ✓ Class-based design for better organization

Usage:
    from json_builder_fixed import DiagramBuilder

    builder = DiagramBuilder(
        bs_connected_path="bs_connected.json",
        compression_results_path="compression_results.json",
        graph_path="graph.json"
    )
    builder.build_diagram()
    builder.save("diagram.json")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DiagramBuilder:
    """Builds enhanced diagram JSON from multiple data sources."""

    def __init__(self, bs_connected_path: str, compression_results_path: str, graph_path: str, scale: int = 16):
        """
        Initialize the diagram builder.

        Args:
            bs_connected_path: Path to bs_connected.json (AI symbol recognition)
            compression_results_path: Path to compression_results.json (Taylor's output)
            graph_path: Path to graph.json (Ryan's connection output)
            scale: Scale factor from compressed grid to original image (default: 16)
        """
        self.bs_connected_path = Path(bs_connected_path)
        self.compression_results_path = Path(compression_results_path)
        self.graph_path = Path(graph_path)
        self.scale = scale

        # Data storage
        self.symbols_data = None  # From bs_connected.json
        self.compression_data = None  # From compression_results.json
        self.graph_data = None  # From graph.json
        self.matrix = None  # Color matrix from compression
        self.diagram = {"symbols": []}  # Output diagram

        # Load all input files
        self._load_data()

    def _load_data(self):
        """Load all required JSON files."""
        # Load symbol data
        if not self.bs_connected_path.exists():
            raise FileNotFoundError(f"bs_connected.json not found: {self.bs_connected_path}")

        with open(self.bs_connected_path) as f:
            self.symbols_data = json.load(f)

        # Load compression data
        if not self.compression_results_path.exists():
            raise FileNotFoundError(f"compression_results.json not found: {self.compression_results_path}")

        with open(self.compression_results_path) as f:
            self.compression_data = json.load(f)

        # Extract matrix from compression data (FIXES UNDEFINED MATRIX ISSUE)
        self._extract_matrix_from_compression()

        # Load graph data
        if not self.graph_path.exists():
            raise FileNotFoundError(f"graph.json not found: {self.graph_path}")

        with open(self.graph_path) as f:
            self.graph_data = json.load(f)

    def _extract_matrix_from_compression(self):
        """
        Extract color matrix from compression results.
        This fixes the original undefined matrix issue.
        """
        if "compressions" not in self.compression_data:
            raise ValueError("compression_results.json missing 'compressions' field")

        # Use the first (smallest) compression level
        compression = self.compression_data["compressions"][0]
        pixels = compression["compressed_pixels"]

        # Build matrix as grid[y][x] = (r, g, b)
        self.matrix = []
        for row in pixels:
            matrix_row = []
            for pixel in row:
                matrix_row.append((pixel["r"], pixel["g"], pixel["b"]))
            self.matrix.append(matrix_row)

    def find_symbol_by_coordinate(self, x: int, y: int) -> Optional[Dict]:
        """
        Find symbol at the given grid coordinate.

        Args:
            x: Grid x-coordinate
            y: Grid y-coordinate

        Returns:
            Symbol dict if found, None otherwise
        """
        # Scale grid coordinates to original image coordinates
        img_x = x * self.scale
        img_y = y * self.scale

        # Search through symbols
        for symbol in self.symbols_data.get("symbols", []):
            x1, y1, x2, y2 = symbol["bbox"]

            # Check if point is inside bounding box
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                return symbol

        return None

    def get_color_at_coordinate(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        """
        Get color at grid coordinate from matrix.

        Args:
            x: Grid x-coordinate
            y: Grid y-coordinate

        Returns:
            (r, g, b) tuple if coordinate is valid, None otherwise
        """
        if self.matrix is None:
            return None

        # Check bounds
        if 0 <= y < len(self.matrix) and 0 <= x < len(self.matrix[y]):
            return self.matrix[y][x]

        return None

    def build_new_symbol(self, symbol: Dict, color: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Build output symbol structure.

        Args:
            symbol: Symbol from bs_connected.json
            color: RGB color tuple (optional)

        Returns:
            Dictionary with tailored symbol info for output
        """
        new_symbol = {
            "name": symbol.get("name", "unknown"),
            "id": symbol.get("id", -1),
            "cls_id": symbol.get("cls_id", -1),
            "bbox": symbol.get("bbox", []),
            "center": symbol.get("center", []),
            "connections": []
        }

        if color is not None:
            new_symbol["color"] = list(color)  # [R, G, B]

        return new_symbol

    def build_diagram(self):
        """
        Build the diagram by integrating data from all sources.

        Uses graph.json as the primary source of symbols and connections,
        then enriches with data from bs_connected.json.
        """
        # Build symbol ID to name mapping from graph
        name_to_symbol = {}

        for node in self.graph_data:
            name = node.get("name", "")
            center = node.get("center", {})

            # Convert center from Ryan's format to grid coordinates
            grid_x = center.get("X", 0)
            grid_y = center.get("Y", 0)

            # Find corresponding symbol from bs_connected.json
            symbol = self.find_symbol_by_coordinate(grid_x, grid_y)

            if symbol:
                # Get color from matrix
                color = self.get_color_at_coordinate(grid_x, grid_y)

                # Build enhanced symbol
                new_symbol = self.build_new_symbol(symbol, color)

                # Store connections (will be filled in next step)
                new_symbol["graph_name"] = name  # Keep graph name for connection mapping
                new_symbol["connections"] = node.get("connections", [])

                name_to_symbol[name] = new_symbol
                self.diagram["symbols"].append(new_symbol)

        # If no symbols found from graph, try building from bs_connected directly
        if not self.diagram["symbols"] and self.symbols_data.get("symbols"):
            print("Warning: No symbols matched from graph. Using bs_connected.json directly.")
            for symbol in self.symbols_data["symbols"]:
                # Estimate grid coordinates from bbox center
                center = symbol.get("center", [0, 0])
                grid_x = center[0] // self.scale
                grid_y = center[1] // self.scale

                color = self.get_color_at_coordinate(grid_x, grid_y)
                new_symbol = self.build_new_symbol(symbol, color)

                self.diagram["symbols"].append(new_symbol)

    def save(self, output_path: str):
        """
        Save the diagram to a JSON file.

        Args:
            output_path: Path to save diagram.json
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.diagram, f, indent=2)

        print(f"Saved diagram with {len(self.diagram['symbols'])} symbols to: {output_path}")


def main():
    """Command-line interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build enhanced diagram JSON from multiple sources"
    )

    parser.add_argument(
        '--bs-connected',
        required=True,
        help='Path to bs_connected.json'
    )

    parser.add_argument(
        '--compression-results',
        required=True,
        help='Path to compression_results.json'
    )

    parser.add_argument(
        '--graph',
        required=True,
        help='Path to graph.json'
    )

    parser.add_argument(
        '--output',
        default='diagram.json',
        help='Path to output diagram.json (default: diagram.json)'
    )

    parser.add_argument(
        '--scale',
        type=int,
        default=16,
        help='Scale factor from grid to image coordinates (default: 16)'
    )

    args = parser.parse_args()

    try:
        builder = DiagramBuilder(
            args.bs_connected,
            args.compression_results,
            args.graph,
            args.scale
        )
        builder.build_diagram()
        builder.save(args.output)

        print("✓ Success")

    except Exception as e:
        print(f"✗ Error: {e}")
        import sys
        sys.exit(1)


if __name__ == '__main__':
    main()
