#!/usr/bin/env python3
"""
Master Pipeline for Pneumatic/Hydraulic Diagram Analysis System.

This script orchestrates all stages of the diagram analysis pipeline:
1. Symbol Detection (Dalton's module) - Detects symbols and creates inpainted image
1.5 Line Detection (Line tracing) - Detects lines using Hough transform on inpainted image
2. Image Compression (Taylor's module) - Compresses symbols + lines into grid
3. Connection Finding (Ryan's module) - Finds connections via BFS on red/green pixels
4. JSON Enhancement (Alden's module) - Combines all data
5. Visualization (Koda's module) - Visualizes the graph

Usage:
    python pipeline.py [OPTIONS] INPUT_PATH

Examples:
    # Auto-detect input type from file extension
    python pipeline.py "0 Input Image/bs.png"

    # Force JSON input mode (skip detection)
    python pipeline.py --mode json bs_connected.json

    # Custom output directory
    python pipeline.py --output-dir /tmp/test_run "0 Input Image/bs.png"
"""

import argparse
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import logging

from pipeline_utils import (
    PipelineError,
    InputValidationError,
    StageExecutionError,
    OutputValidationError,
    PipelineConfig,
    InputValidator,
    OutputValidator
)


class DiagramPipeline:
    """Main pipeline orchestrator."""

    def __init__(self, input_path, output_dir=None, mode='auto', verbose=False):
        """
        Initialize the pipeline.

        Args:
            input_path: Path to input file (image or JSON)
            output_dir: Output directory (default: timestamped dir in 6 Output Image/)
            mode: Input mode ('auto', 'image', or 'json')
            verbose: Enable verbose output
        """
        self.input_path = Path(input_path)
        self.mode = mode
        self.verbose = verbose

        # Set up logging
        self.logger = self._setup_logging(verbose)

        # Determine input type
        self.input_type = self._determine_input_type()

        # Create output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = PipelineConfig.create_output_dir(timestamp)

        self.logger.info(f"Output directory: {self.output_dir}")

        # Track which stages to run
        self.run_detection = (self.input_type == 'image')

    def _setup_logging(self, verbose):
        """Set up logging configuration."""
        logger = logging.getLogger('pipeline')
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def _determine_input_type(self):
        """Determine whether input is image or JSON."""
        if self.mode == 'image':
            return 'image'
        elif self.mode == 'json':
            return 'json'
        else:  # auto mode
            _, file_type = InputValidator.validate_input_file(self.input_path)
            return file_type

    def run(self):
        """Execute the full pipeline."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("PNEUMATIC/HYDRAULIC DIAGRAM ANALYSIS PIPELINE")
            self.logger.info("=" * 60)
            self.logger.info(f"Input: {self.input_path}")
            self.logger.info(f"Mode: {self.input_type}")
            self.logger.info("")

            # Validate project structure
            errors = PipelineConfig.validate_project_structure()
            if errors:
                raise PipelineError("Project structure validation failed:\n" + "\n".join(errors))

            # Stage 1: Symbol Detection (if needed)
            if self.run_detection:
                bs_connected_path, inpainted_path = self._stage_1_detection()
            else:
                # Copy input JSON to output directory
                bs_connected_path = self.output_dir / PipelineConfig.BS_CONNECTED_JSON
                self.logger.info("Stage 1: Symbol Detection [SKIPPED - using provided JSON]")
                shutil.copy(self.input_path, bs_connected_path)
                InputValidator.validate_bs_connected_json(bs_connected_path)
                inpainted_path = None

            # Stage 1.5: Line Detection (if we have an inpainted image)
            if inpainted_path and inpainted_path.exists():
                lines_path = self._stage_1_5_line_detection(inpainted_path)
            else:
                # Check if the input JSON already contains lines
                lines_path = self._extract_lines_from_json(bs_connected_path)

            # Stage 2: Image Compression
            compression_results_path = self._stage_2_compression(bs_connected_path, lines_path)

            # Stage 3: Connection Finding
            graph_path, ryan_output_png = self._stage_3_connections(compression_results_path)

            # Stage 4: JSON Enhancement
            diagram_path = self._stage_4_json_builder(
                bs_connected_path,
                compression_results_path,
                graph_path
            )

            # Stage 5: Visualization
            graph_viz_path = self._stage_5_visualization(graph_path)

            # Pipeline complete
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info("")
            self.logger.info("Generated files:")
            self.logger.info(f"  - {PipelineConfig.BS_CONNECTED_JSON}")
            if (self.output_dir / PipelineConfig.INPAINTED_PNG).exists():
                self.logger.info(f"  - {PipelineConfig.INPAINTED_PNG}")
            if (self.output_dir / PipelineConfig.LINES_JSON).exists():
                self.logger.info(f"  - {PipelineConfig.LINES_JSON}")
            self.logger.info(f"  - {PipelineConfig.COMPRESSION_RESULTS_JSON}")
            self.logger.info(f"  - {PipelineConfig.GRAPH_JSON}")
            self.logger.info(f"  - {PipelineConfig.OUTPUT_PNG}")
            self.logger.info(f"  - {PipelineConfig.DIAGRAM_JSON}")
            self.logger.info(f"  - {PipelineConfig.GRAPH_VISUALIZED_PNG}")

            return True

        except PipelineError as e:
            self.logger.error(f"\nPIPELINE FAILED: {e}")
            return False
        except Exception as e:
            self.logger.error(f"\nUNEXPECTED ERROR: {e}", exc_info=True)
            return False

    def _stage_1_detection(self):
        """Stage 1: Symbol Detection using Dalton's module."""
        self.logger.info("Stage 1: Symbol Detection")
        self.logger.info("-" * 60)

        output_path = self.output_dir / PipelineConfig.BS_CONNECTED_JSON
        inpainted_path = self.output_dir / PipelineConfig.INPAINTED_PNG

        try:
            # Run batch detection script
            cmd = [
                sys.executable,
                str(PipelineConfig.BATCH_DETECT_SCRIPT),
                str(self.input_path),
                "--output",
                str(output_path),
                "--inpainted",
                str(inpainted_path)
            ]

            self.logger.info(f"  Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                check=True
            )

            if self.verbose and result.stdout:
                self.logger.debug(result.stdout)

            # Validate output
            OutputValidator.validate_file_exists(output_path, "Symbol detection output")
            OutputValidator.validate_json_file(output_path, required_fields=["symbols"])

            self.logger.info(f"  ✓ Output: {output_path.name}")
            if inpainted_path.exists():
                self.logger.info(f"  ✓ Output: {inpainted_path.name}")
            self.logger.info("")

            return output_path, inpainted_path

        except subprocess.CalledProcessError as e:
            raise StageExecutionError(
                "Symbol Detection",
                f"Detection script failed: {e.stderr if e.stderr else str(e)}",
                e
            )
        except Exception as e:
            raise StageExecutionError("Symbol Detection", str(e), e)

    def _extract_lines_from_json(self, bs_connected_path):
        """Extract lines from bs_connected.json if they exist."""
        import json

        try:
            with open(bs_connected_path, 'r') as f:
                data = json.load(f)

            lines = data.get('lines', [])

            if lines:
                # Save lines to separate file
                lines_path = self.output_dir / PipelineConfig.LINES_JSON
                with open(lines_path, 'w') as f:
                    json.dump({"lines": lines}, f, indent=2)

                self.logger.info("Stage 1.5: Line Detection [SKIPPED - using lines from input JSON]")
                self.logger.info(f"  Found {len(lines)} lines in input JSON")
                self.logger.info("")
                return lines_path
            else:
                self.logger.info("Stage 1.5: Line Detection [SKIPPED - no lines in input JSON]")
                self.logger.info("")
                return None

        except Exception as e:
            self.logger.warning(f"Could not extract lines from JSON: {e}")
            return None

    def _stage_1_5_line_detection(self, inpainted_path):
        """Stage 1.5: Line Detection using line tracing."""
        self.logger.info("Stage 1.5: Line Detection")
        self.logger.info("-" * 60)

        output_path = self.output_dir / PipelineConfig.LINES_JSON

        try:
            # Run line detection script
            cmd = [
                sys.executable,
                str(PipelineConfig.LINE_TRACE_SCRIPT),
                str(inpainted_path),
                "--output",
                str(output_path)
            ]

            self.logger.info(f"  Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                check=True
            )

            if self.verbose and result.stdout:
                self.logger.debug(result.stdout)

            # Validate output
            OutputValidator.validate_file_exists(output_path, "Line detection output")
            OutputValidator.validate_json_file(output_path, required_fields=["lines"])

            self.logger.info(f"  ✓ Output: {output_path.name}")
            self.logger.info("")

            return output_path

        except subprocess.CalledProcessError as e:
            raise StageExecutionError(
                "Line Detection",
                f"Line detection script failed: {e.stderr if e.stderr else str(e)}",
                e
            )
        except Exception as e:
            raise StageExecutionError("Line Detection", str(e), e)

    def _stage_2_compression(self, bs_connected_path, lines_path=None):
        """Stage 2: Image Compression using Taylor's module."""
        self.logger.info("Stage 2: Image Compression")
        self.logger.info("-" * 60)

        try:
            # Run compression script
            cmd = [
                sys.executable,
                str(PipelineConfig.COMPRESSION_SCRIPT),
                "--input-json",
                str(bs_connected_path),
                "--output-dir",
                str(self.output_dir)
            ]

            # Add lines JSON if available
            if lines_path and lines_path.exists():
                cmd.extend(["--lines-json", str(lines_path)])

            self.logger.info(f"  Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                check=True,
                cwd=str(PipelineConfig.MODULE_2_DIR)
            )

            if self.verbose and result.stdout:
                self.logger.debug(result.stdout)

            # Validate output
            output_path = self.output_dir / PipelineConfig.COMPRESSION_RESULTS_JSON
            OutputValidator.validate_file_exists(output_path, "Compression results")
            OutputValidator.validate_json_file(output_path, required_fields=["compressions"])

            self.logger.info(f"  ✓ Output: {output_path.name}")
            self.logger.info("")

            return output_path

        except subprocess.CalledProcessError as e:
            raise StageExecutionError(
                "Image Compression",
                f"Compression script failed: {e.stderr if e.stderr else str(e)}",
                e
            )
        except Exception as e:
            raise StageExecutionError("Image Compression", str(e), e)

    def _stage_3_connections(self, compression_results_path):
        """Stage 3: Connection Finding using Ryan's module."""
        self.logger.info("Stage 3: Connection Finding")
        self.logger.info("-" * 60)

        ryan_input_path = self.output_dir / PipelineConfig.RYAN_INPUT_JSON
        graph_output_path = self.output_dir / PipelineConfig.GRAPH_JSON
        viz_output_path = self.output_dir / PipelineConfig.OUTPUT_PNG

        try:
            # First, convert format using adapter
            # Use compression index 2 (finest 8x8 grid) for better symbol preservation
            self.logger.info("  Converting format (Taylor → Ryan)...")
            cmd_adapter = [
                sys.executable,
                str(PipelineConfig.FORMAT_ADAPTER_SCRIPT),
                str(compression_results_path),
                str(ryan_input_path),
                "--compression-index", "2"  # Use finest grid (8x8)
            ]

            subprocess.run(cmd_adapter, capture_output=True, text=True, check=True)

            # Run connection finding script
            cmd_connect = [
                sys.executable,
                str(PipelineConfig.CONNECTION_SCRIPT),
                str(ryan_input_path),
                "--output-dir",
                str(self.output_dir)
            ]

            self.logger.info(f"  Running: {' '.join(cmd_connect)}")

            result = subprocess.run(
                cmd_connect,
                capture_output=not self.verbose,
                text=True,
                check=True
            )

            if self.verbose and result.stdout:
                self.logger.debug(result.stdout)

            # Validate outputs
            OutputValidator.validate_file_exists(graph_output_path, "Graph JSON")
            OutputValidator.validate_json_file(graph_output_path)

            self.logger.info(f"  ✓ Output: {graph_output_path.name}")
            if viz_output_path.exists():
                self.logger.info(f"  ✓ Output: {viz_output_path.name}")
            self.logger.info("")

            return graph_output_path, viz_output_path

        except subprocess.CalledProcessError as e:
            raise StageExecutionError(
                "Connection Finding",
                f"Connection script failed: {e.stderr if e.stderr else str(e)}",
                e
            )
        except Exception as e:
            raise StageExecutionError("Connection Finding", str(e), e)

    def _stage_4_json_builder(self, bs_connected_path, compression_results_path, graph_path):
        """Stage 4: JSON Enhancement using Alden's module."""
        self.logger.info("Stage 4: JSON Enhancement")
        self.logger.info("-" * 60)

        output_path = self.output_dir / PipelineConfig.DIAGRAM_JSON

        try:
            # Import and use the JSON builder module
            import sys
            sys.path.insert(0, str(PipelineConfig.MODULE_4_DIR))

            from json_builder_fixed import DiagramBuilder

            self.logger.info("  Building enhanced diagram JSON...")

            builder = DiagramBuilder(
                str(bs_connected_path),
                str(compression_results_path),
                str(graph_path)
            )
            builder.build_diagram()
            builder.save(str(output_path))

            # Validate output
            OutputValidator.validate_file_exists(output_path, "Diagram JSON")
            OutputValidator.validate_json_file(output_path, required_fields=["symbols"])

            self.logger.info(f"  ✓ Output: {output_path.name}")
            self.logger.info("")

            return output_path

        except ImportError as e:
            raise StageExecutionError(
                "JSON Enhancement",
                f"Failed to import json_builder_fixed: {e}. Make sure the module is created.",
                e
            )
        except Exception as e:
            raise StageExecutionError("JSON Enhancement", str(e), e)

    def _stage_5_visualization(self, graph_path):
        """Stage 5: Visualization using Koda's module."""
        self.logger.info("Stage 5: Visualization")
        self.logger.info("-" * 60)

        output_path = self.output_dir / PipelineConfig.GRAPH_VISUALIZED_PNG

        try:
            # Run visualizer script
            cmd = [
                sys.executable,
                str(PipelineConfig.VISUALIZER_SCRIPT),
                "--graph-json",
                str(graph_path),
                "--output",
                str(output_path)
            ]

            self.logger.info(f"  Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                check=True
            )

            if self.verbose and result.stdout:
                self.logger.debug(result.stdout)

            # Validate output
            OutputValidator.validate_file_exists(output_path, "Graph visualization")

            self.logger.info(f"  ✓ Output: {output_path.name}")
            self.logger.info("")

            return output_path

        except subprocess.CalledProcessError as e:
            raise StageExecutionError(
                "Visualization",
                f"Visualizer script failed: {e.stderr if e.stderr else str(e)}",
                e
            )
        except Exception as e:
            raise StageExecutionError("Visualization", str(e), e)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pneumatic/Hydraulic Diagram Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect input type from file extension
  python pipeline.py "0 Input Image/bs.png"

  # Force JSON input mode (skip detection)
  python pipeline.py --mode json bs_connected.json

  # Custom output directory
  python pipeline.py --output-dir /tmp/test_run "0 Input Image/bs.png"
        """
    )

    parser.add_argument(
        'input_path',
        help='Path to input file (image or bs_connected.json)'
    )

    parser.add_argument(
        '--mode',
        choices=['auto', 'image', 'json'],
        default='auto',
        help='Input mode: auto (detect from extension), image (force detection), json (skip detection)'
    )

    parser.add_argument(
        '--output-dir',
        help='Output directory (default: timestamped dir in "6 Output Image/")'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = DiagramPipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        mode=args.mode,
        verbose=args.verbose
    )

    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
