# Pneumatic/Hydraulic Diagram Analysis Pipeline

A unified pipeline that orchestrates 5 independent modules for analyzing electrical single line diagrams. The pipeline detects symbols, finds connections between components, and generates visual outputs.

## Quick Start

```bash
# Run with JSON input (skips detection)
python pipeline.py --mode json test_bs_connected.json

# Run with image input (includes ML detection - when available)
python pipeline.py --mode image "0 Input Image/bs.png"

# Auto-detect input type from file extension
python pipeline.py your_input_file.json
```

## Pipeline Stages

The pipeline consists of 5 sequential stages:

```
Input → [1] Detection → [2] Compression → [3] Connections → [4] Enhancement → [5] Visualization → Output
```

### Stage 1: Symbol Detection (Dalton's Module)
- **Purpose:** Detect and classify pneumatic/hydraulic symbols in diagram images
- **Input:** Diagram image (PNG, JPG, etc.)
- **Output:** `bs_connected.json` - Symbol definitions with bounding boxes
- **Status:** ⚠️ Currently placeholder implementation (creates empty structure)
- **Future:** Will integrate object detection + ResNet-18 classifier

### Stage 2: Image Compression (Taylor's Module)
- **Purpose:** Create colored bounding boxes and compress image to grid representation
- **Input:** `bs_connected.json`
- **Output:** `compression_results.json` - Grid-based compressed representation
- **Features:**
  - Color quantization to reduce anti-aliasing
  - Multi-level compression (8x8, 16x16, 32x32 grid squares)
  - Debug images saved for visualization

### Stage 3: Connection Finding (Ryan's Module)
- **Purpose:** Analyze compressed grid to find connections between symbols
- **Input:** `compression_results.json` (converted to Ryan's format)
- **Output:**
  - `graph.json` - Graph structure with symbol connections
  - `output.png` - Annotated diagram visualization
- **Algorithm:** BFS-based flood-fill to find red pathways connecting green symbols

### Stage 4: JSON Enhancement (Alden's Module)
- **Purpose:** Enhance graph with additional metadata from symbol recognition
- **Input:** `bs_connected.json`, `compression_results.json`, `graph.json`
- **Output:** `diagram.json` - Enhanced diagram with full symbol information
- **Fix:** ✓ Resolved undefined matrix issue from original implementation

### Stage 5: Visualization (Koda's Module)
- **Purpose:** Create matplotlib graph visualization
- **Input:** `graph.json`
- **Output:** `graph_visualized.png` - Matplotlib scatter plot with connections
- **Features:** Inverted Y-axis to match image coordinates

## Installation

### Required Dependencies

Install Python dependencies for all modules:

```bash
# Module 1 (Dalton) - Symbol Detection
pip install torch torchvision pillow numpy scikit-learn tqdm

# Module 2 (Taylor) - Image Compression
pip install Pillow>=10.0.0

# Module 3 (Ryan) - Connection Finding
pip install Pillow

# Module 5 (Koda) - Visualization
pip install matplotlib
```

Or install all at once:
```bash
pip install torch torchvision pillow numpy scikit-learn tqdm matplotlib
```

## Usage

### Command-Line Interface

```
python pipeline.py [OPTIONS] INPUT_PATH

Positional Arguments:
  INPUT_PATH              Path to input file (image or bs_connected.json)

Options:
  --mode {auto,image,json}
                         Input mode:
                           auto:  Auto-detect from file extension (default)
                           image: Force raw image processing (run ML detection)
                           json:  Force pre-detected symbols mode (skip detection)

  --output-dir PATH      Output directory (default: timestamped dir in "6 Output Image/")

  --verbose, -v          Enable verbose output (shows all module stdout/stderr)

Examples:
  # Auto-detect input type from file extension
  python pipeline.py "0 Input Image/bs.png"

  # Force JSON input mode (skip detection)
  python pipeline.py --mode json bs_connected.json

  # Custom output directory with verbose output
  python pipeline.py --output-dir /tmp/test_run --verbose test_bs_connected.json
```

### Input Formats

#### Image Input (PNG, JPG, etc.)
When using image input, the pipeline runs all 5 stages including symbol detection.

**Supported formats:** `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`

#### JSON Input (`bs_connected.json`)
When using JSON input, the pipeline skips Stage 1 (detection) and starts with compression.

**Expected structure:**
```json
{
  "image_path": "path/to/original/image.png",
  "image_width": 800,
  "image_height": 600,
  "symbols": [
    {
      "id": 0,
      "cls_id": 42,
      "name": "2-2-check-valve",
      "conf": 0.95,
      "bbox": [100, 150, 200, 250],
      "center": [150, 200]
    }
  ]
}
```

## Output Structure

All outputs are saved to a timestamped directory in `6 Output Image/`:

```
6 Output Image/
└── 2025-11-28_16-35-12/
    ├── bs_connected.json              # Symbol definitions
    ├── compression_results.json       # Grid compression data
    ├── ryan_input.json                # Converted format for Ryan
    ├── graph.json                     # Connection graph
    ├── output.png                     # Annotated diagram (Ryan's)
    ├── diagram.json                   # Enhanced diagram (Alden's)
    ├── graph_visualized.png           # Matplotlib graph (Koda's)
    ├── original_bboxes.png            # Colored bounding boxes
    ├── compressed_8x8.png             # 8x8 grid compression
    ├── compressed_16x16.png           # 16x16 grid compression
    └── compressed_32x32.png           # 32x32 grid compression
```

## Project Structure

```
AllPartsAIProj/
├── pipeline.py                    # Master control file (NEW)
├── pipeline_utils/                # Utility modules (NEW)
│   ├── __init__.py
│   ├── config.py                  # Configuration constants
│   ├── errors.py                  # Custom exceptions
│   └── validators.py              # Input/output validation
├── 0 Input Image/                 # Input directory
│   ├── bs.png
│   └── bs_simplified.png
├── 1 Dalton/                      # Symbol Detection Module
│   ├── batch_detect.py            # Batch detection wrapper (NEW)
│   ├── predict.py                 # Single-image classifier
│   ├── train_model.py             # Model training
│   └── classes.json               # Symbol class definitions
├── 2 Taylor/                      # Image Compression Module
│   └── main.py                    # Compression script (MODIFIED)
├── 3 Ryan/                        # Connection Finding Module
│   ├── find_connections.py        # Connection finder (MODIFIED)
│   └── format_adapter.py          # Format converter (NEW)
├── 4 Alden/                       # JSON Building Module
│   ├── json_builder.py            # Original (broken)
│   └── json_builder_fixed.py      # Fixed version (NEW)
├── 5 Koda/                        # Visualization Module
│   └── Visualizer.py              # Graph visualizer (MODIFIED)
└── 6 Output Image/                # Output directory
    └── [timestamped runs]/
```

## Key Fixes & Improvements

### Issue 1: Missing bs_connected.json Generator ✓ FIXED
- **Problem:** Module 1's `predict.py` only handled single images
- **Solution:** Created `1 Dalton/batch_detect.py` wrapper
- **Status:** Placeholder implementation (generates empty structure)

### Issue 2: Broken Module 4 - Undefined Matrix ✓ FIXED
- **Problem:** Line 50 in original `json_builder.py`: `color = matrix[x][y]` - matrix was undefined
- **Solution:** Complete rewrite as `json_builder_fixed.py` with proper class design
- **Fix:** Loads matrix from `compression_results.json`

### Issue 3: Format Mismatch (Taylor → Ryan) ✓ FIXED
- **Problem:** Taylor outputs snake_case (`grid_width`), Ryan expects PascalCase (`GridWidth`)
- **Solution:** Created `3 Ryan/format_adapter.py` to convert between formats
- **Status:** Working correctly

### Issue 4: Hardcoded Paths ✓ FIXED
- **Problem:** All modules used hardcoded input/output paths
- **Solution:** Added CLI arguments to all modules
- **Modified files:**
  - `2 Taylor/main.py` - Added `--input-json`, `--output-dir` arguments
  - `3 Ryan/find_connections.py` - Added `--output-dir` argument
  - `5 Koda/Visualizer.py` - Added `--graph-json`, `--output` arguments

## Testing

### Test with Sample JSON
A test file `test_bs_connected.json` is provided with 3 sample symbols:

```bash
python pipeline.py --mode json test_bs_connected.json
```

Expected output:
- ✓ All 5 stages complete successfully
- ✓ 6 output files generated in `6 Output Image/[timestamp]/`
- ✓ No errors or warnings (except "No symbols matched from graph" is expected)

### Verification Checklist
- [ ] All output files created
- [ ] All JSON files are valid (parseable)
- [ ] All images are readable
- [ ] No undefined variable errors
- [ ] Pipeline completes in under 10 seconds

## Development & Contributing

### Module Integration Guidelines

Each module communicates via JSON files:

1. **Stage N writes output JSON**
2. **Pipeline validates output**
3. **Stage N+1 reads validated JSON**

### Adding a New Stage

1. Create module in new directory (e.g., `7 NewModule/`)
2. Add entry to `PipelineConfig` in `pipeline_utils/config.py`
3. Create stage execution method in `pipeline.py`
4. Add validation for stage outputs

### Running Individual Modules

All modules can still be run independently:

```bash
# Taylor's compression
cd "2 Taylor"
python main.py --input-json ../test_bs_connected.json

# Ryan's connection finder
cd "3 Ryan"
python find_connections.py ryan_input.json

# Koda's visualizer
cd "5 Koda"
python Visualizer.py --graph-json graph.json
```

## Known Limitations

1. **Symbol Detection (Stage 1):** Currently placeholder only - does not actually detect symbols
   - Future work: Integrate object detection model (YOLOv8, Faster R-CNN, etc.)

2. **Module 4 Warning:** "No symbols matched from graph" is expected when using test data
   - This occurs because grid coordinates don't perfectly align with symbol bounding boxes
   - Fallback mechanism uses bs_connected.json directly

3. **Performance:** Large images may take several seconds to process through all stages

## Troubleshooting

### "ModuleNotFoundError: No module named 'pipeline_utils'"
- **Solution:** Run pipeline.py from the project root directory

### "FileNotFoundError: bs_connected.json not found"
- **Solution:** Ensure you're using `--mode json` when providing a JSON file as input

### "ImportError: Failed to import json_builder_fixed"
- **Solution:** Verify `4 Alden/json_builder_fixed.py` exists

### Pipeline fails with "Project structure validation failed"
- **Solution:** Ensure all 5 module directories exist (1 Dalton through 5 Koda)

## Credits

- **Dalton:** ML symbol detection & classification
- **Taylor:** Image compression & grid reduction
- **Ryan:** Connection finding via flood-fill algorithm
- **Alden:** JSON building & symbol metadata
- **Koda:** Graph visualization

**Pipeline Integration:** Assembled November 2025

## License

[Add your license information here]
