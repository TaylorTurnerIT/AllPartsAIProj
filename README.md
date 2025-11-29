# 🔌 Pneumatic/Hydraulic Diagram Analysis Pipeline

> **Automated detection, line tracing, and connection analysis for pneumatic and hydraulic single-line diagrams**

A complete end-to-end pipeline that detects symbols, traces connecting lines, analyzes component relationships, and generates visual connection graphs from schematic diagrams.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Pipeline Architecture](#-pipeline-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline Stages](#-pipeline-stages)
- [Output Files](#-output-files)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Development](#-development)

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment (Python 3.11 recommended)
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Run the pipeline
python pipeline.py "1 Dalton/bs.png"
```

**That's it!** The pipeline will:
- ✅ Detect symbols using YOLO (transformers, breakers)
- ✅ Remove symbols and detect connecting lines via Hough transform
- ✅ Compress to a grid representation (green=symbols, red=lines)
- ✅ Find connections using BFS through the grid
- ✅ Generate connection graph and visualizations

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: Diagram Image                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  STAGE 1: Symbol Detection (YOLO)       │
        │  • Multi-orientation detection           │
        │  • Detects transformers & breakers       │
        │  • Outputs: symbols JSON + inpainted PNG │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  STAGE 1.5: Line Detection (Hough)      │
        │  • Canny edge detection                  │
        │  • Hough line transform                  │
        │  • Outputs: lines JSON                   │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  STAGE 2: Image Compression              │
        │  • Draw symbols (GREEN pixels)           │
        │  • Draw lines (RED pixels)               │
        │  • Compress to 8x8 grid                  │
        │  • Outputs: compressed grid JSON         │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  STAGE 3: Connection Finding (BFS)       │
        │  • BFS from green through red pixels     │
        │  • Finds which symbols connect           │
        │  • Outputs: connection graph JSON        │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  STAGE 4: JSON Enhancement               │
        │  • Combines all data sources             │
        │  • Adds metadata                         │
        │  • Outputs: enhanced diagram JSON        │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  STAGE 5: Visualization                  │
        │  • Generates matplotlib graph            │
        │  • Shows connections and labels          │
        │  • Outputs: PNG visualization            │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT: Graphs, JSONs, Visualizations               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation

### Prerequisites

- **Python 3.11** (recommended for stability)
- **uv** package manager (recommended) or pip

### Install uv (Recommended)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup Environment

```bash
# 1. Create virtual environment with Python 3.11
uv venv --python 3.11

# 2. Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install all dependencies
uv pip install -r requirements.txt
```

### Alternative: Using pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Required Dependencies

- **OpenCV** (`opencv-python-headless`) - Line detection, image processing
- **NumPy** - Array operations
- **Pillow** - Image manipulation
- **Ultralytics** - YOLO object detection
- **PyTorch** - Deep learning backend
- **Matplotlib** - Visualization

See [`requirements.txt`](requirements.txt) for complete list.

---

## 🎯 Usage

### Basic Usage

```bash
# Run on an image (full pipeline)
python pipeline.py "path/to/diagram.png"

# Run on existing JSON (skip detection)
python pipeline.py --mode json "bs_connected.json"
```

### Command-Line Options

```
python pipeline.py [OPTIONS] INPUT_PATH

Positional Arguments:
  INPUT_PATH              Path to input file (image or JSON)

Options:
  --mode {auto,image,json}
                         Input mode:
                           auto:  Auto-detect from extension (default)
                           image: Force image detection
                           json:  Skip detection, use provided JSON

  --output-dir PATH      Custom output directory
                         (default: timestamped in "6 Output Image/")

  --verbose, -v          Show detailed logs from all modules
```

### Examples

```bash
# Auto-detect input type
python pipeline.py "1 Dalton/bs.png"

# Force JSON mode (skip detection)
python pipeline.py --mode json test_bs_connected.json

# Custom output directory with verbose logging
python pipeline.py --output-dir ./results --verbose "diagram.png"

# Use a different image
python pipeline.py "0 Input Image/bs_simplified.png"
```

---

## 🔄 Pipeline Stages

### Stage 1: Symbol Detection
**Module:** `1 Dalton/run_new_best.py`

Detects pneumatic/hydraulic symbols using YOLO object detection with multi-orientation processing.

**Key Features:**
- Multi-orientation detection (0°, 90°, 270°) to catch vertical components
- Detects transformers and breakers
- Box merging and deduplication
- Confidence filtering and top-K selection

**Inputs:**
- Diagram image (PNG, JPG, etc.)
- YOLO model (`new_best.pt`)

**Outputs:**
- `bs_connected.json` - Detected symbols with bounding boxes
- `inpainted.png` - Image with symbols removed (for line detection)

**Configuration:**
```python
KEEP_TOP_TRANSFORMERS = 2
KEEP_TOP_BREAKERS = 24
BOX_SHRINK_BREAKER = 0.85
BOX_SHRINK_TRANSFORMER = 0.8
```

---

### Stage 1.5: Line Detection
**Module:** `1 Dalton/line_trace.py`

Detects connecting lines using Hough transform on the inpainted image.

**Algorithm:**
1. Binarize and invert image (lines → white on black)
2. Apply Gaussian blur for noise reduction
3. Canny edge detection
4. Morphological skeletonization
5. Probabilistic Hough Line Transform

**Inputs:**
- `inpainted.png` (symbols removed)

**Outputs:**
- `lines.json` - Line segments with endpoints and length

**Configuration:**
```python
HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 20
GAUSS_BLUR = 3
```

---

### Stage 2: Image Compression
**Module:** `2 Taylor/main.py`

Creates a color-coded representation and compresses to grid format.

**Color System:**
- **GREEN** `(0, 255, 0)` → Symbols (where G > R and G > B)
- **RED** `(255, 0, 0)` → Lines (where R > G and R > B)
- **WHITE** `(255, 255, 255)` → Background

**Process:**
1. Draw detected lines as **8px thick red** lines
2. Draw symbol bounding boxes as **solid green** rectangles
3. Compress to grid (8x8, 16x16, 32x32 squares)
4. **Green pixels have priority** over red during compression

**Inputs:**
- `bs_connected.json` (symbols)
- `lines.json` (lines)

**Outputs:**
- `compression_results.json` - Multi-level grid compression
- `original_bboxes.png` - Colored symbol/line image
- `compressed_*.png` - Grid visualizations

**Grid Levels:**
```
Level 0: 32x32 squares (coarse)
Level 1: 16x16 squares (medium)
Level 2: 8x8 squares (fine) ← Used for connection finding
```

---

### Stage 3: Connection Finding
**Module:** `3 Ryan/find_connections.py`

Finds connections between symbols using BFS through the compressed grid.

**Algorithm:**
1. Identify **glyphs** (contiguous green regions) via BFS
2. For each glyph, BFS through **red pixels** (lines)
3. Record which other glyphs are reached
4. Generate bidirectional connection graph

**Color Detection:**
```python
is_green(r, g, b) = g > r and g > b  # Symbols
is_red(r, g, b)   = r > g and r > b  # Lines
```

**Inputs:**
- `ryan_input.json` (converted from compression_results.json)

**Outputs:**
- `graph.json` - Connection graph with glyph names
- `output.png` - Annotated visualization with labels

**Example Graph:**
```json
[
  {
    "name": "A",
    "connections": ["B", "E"],
    "center": {"X": 18, "Y": 35}
  },
  {
    "name": "B",
    "connections": ["A"],
    "center": {"X": 67, "Y": 35}
  }
]
```

---

### Stage 4: JSON Enhancement
**Module:** `4 Alden/json_builder_fixed.py`

Combines all data sources and enriches the diagram metadata.

**Inputs:**
- `bs_connected.json` (original symbols)
- `compression_results.json` (grid data)
- `graph.json` (connections)

**Outputs:**
- `diagram.json` - Complete diagram with all metadata

---

### Stage 5: Visualization
**Module:** `5 Koda/Visualizer.py`

Generates a matplotlib graph showing symbol positions and connections.

**Features:**
- Scatter plot of symbol centers
- Arrows showing connections
- Labels for each node
- Inverted Y-axis to match image coordinates

**Inputs:**
- `graph.json`

**Outputs:**
- `graph_visualized.png`

---

## 📂 Output Files

All outputs are saved to timestamped directories in `6 Output Image/`:

```
6 Output Image/
└── 2025-11-28_18-45-30/
    ├── bs_connected.json           # Detected symbols
    ├── inpainted.png               # Image with symbols removed
    ├── lines.json                  # Detected line segments
    ├── compression_results.json    # Grid compression (all levels)
    ├── original_bboxes.png         # Green symbols + red lines
    ├── compressed_8x8.png          # Finest grid compression
    ├── compressed_16x16.png        # Medium grid compression
    ├── compressed_32x32.png        # Coarsest grid compression
    ├── ryan_input.json             # Format-converted grid
    ├── graph.json                  # Connection graph
    ├── output.png                  # Ryan's labeled visualization
    ├── diagram.json                # Enhanced diagram metadata
    └── graph_visualized.png        # Matplotlib graph
```

### Key Output Files

| File | Description |
|------|-------------|
| `bs_connected.json` | Symbol detections with bboxes, class IDs, confidence |
| `lines.json` | Line segments with endpoints (x1, y1, x2, y2) |
| `graph.json` | Connection graph (which symbols connect) |
| `diagram.json` | Complete diagram with all metadata |
| `original_bboxes.png` | Visual: green boxes + red lines |
| `output.png` | Visual: compressed grid with labels |
| `graph_visualized.png` | Visual: matplotlib connection graph |

---

## ⚙️ Configuration

### Pipeline Settings

Edit `pipeline_utils/config.py` to customize:

```python
# Compression settings
DEFAULT_COMPRESSION_SIZE = 16
COMPRESSION_SIZES = [8, 16, 32]  # Grid square sizes

# Symbol detection
CONFIDENCE_THRESHOLD = 0.5

# Color quantization
ENABLE_COLOR_QUANTIZATION = True
COLOR_THRESHOLD = 500.0
```

### Line Detection Parameters

Edit `1 Dalton/line_trace.py`:

```python
HOUGH_THRESHOLD = 40        # Minimum votes for line
HOUGH_MIN_LINE_LENGTH = 20  # Minimum line length (pixels)
HOUGH_MAX_LINE_GAP = 20     # Max gap to bridge (pixels)
GAUSS_BLUR = 3              # Blur kernel size
```

### Symbol Detection Parameters

Edit `1 Dalton/run_new_best.py`:

```python
KEEP_TOP_TRANSFORMERS = 2   # Max transformer detections
KEEP_TOP_BREAKERS = 24      # Max breaker detections
BOX_SHRINK_BREAKER = 0.85   # Shrink breaker boxes to 85%
BOX_SHRINK_TRANSFORMER = 0.8  # Shrink transformer boxes to 80%
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:** Install dependencies
```bash
uv pip install -r requirements.txt
```

#### 2. `FileNotFoundError: new_best.pt not found`

**Solution:** Ensure the YOLO model exists
```bash
ls "1 Dalton/new_best.pt"
```

#### 3. Pipeline fails at Stage 1 with "Module 'cv2' not found"

**Solution:** Install OpenCV
```bash
uv pip install opencv-python-headless
```

#### 4. Too few glyphs detected (symbols merging together)

**Cause:** Grid is too coarse, symbols are merging

**Solution:** Pipeline already uses finest 8x8 grid. If still an issue:
1. Increase line width in `2 Taylor/main.py`: `line_width: int = 8`
2. Ensure green priority is enabled (already default)

#### 5. No connections found between symbols

**Possible causes:**
- Lines too thin (increase `line_width` in Stage 2)
- Hough parameters too strict (lower `HOUGH_THRESHOLD`)
- Symbols not detected (check `bs_connected.json`)

**Debug steps:**
```bash
# Check detected symbols
cat "6 Output Image/[timestamp]/bs_connected.json" | grep "id"

# Check detected lines
cat "6 Output Image/[timestamp]/lines.json" | grep "length"

# View intermediate images
open "6 Output Image/[timestamp]/original_bboxes.png"
open "6 Output Image/[timestamp]/compressed_8x8.png"
```

---

## 🛠️ Development

### Project Structure

```
AllPartsAIProj/
├── pipeline.py                    # Main pipeline orchestrator
├── requirements.txt               # Python dependencies
├── pipeline_utils/                # Shared utilities
│   ├── __init__.py
│   ├── config.py                  # Configuration constants
│   ├── errors.py                  # Custom exceptions
│   └── validators.py              # Input/output validation
│
├── 1 Dalton/                      # Symbol Detection Module
│   ├── run_new_best.py            # YOLO detection script
│   ├── line_trace.py              # Hough line detection
│   ├── new_best.pt                # YOLO model weights
│   └── classes.json               # Symbol class definitions
│
├── 2 Taylor/                      # Image Compression Module
│   └── main.py                    # Grid compression
│
├── 3 Ryan/                        # Connection Finding Module
│   ├── find_connections.py        # BFS connection finder
│   └── format_adapter.py          # Format converter
│
├── 4 Alden/                       # JSON Enhancement Module
│   └── json_builder_fixed.py     # Metadata enrichment
│
├── 5 Koda/                        # Visualization Module
│   └── Visualizer.py              # Graph visualizer
│
└── 6 Output Image/                # Output directory
    └── [timestamped runs]/
```

### Running Individual Modules

Each module can be run standalone for testing:

```bash
# Symbol detection
python "1 Dalton/run_new_best.py" "image.png" --output "symbols.json" --inpainted "inpainted.png"

# Line detection
python "1 Dalton/line_trace.py" "inpainted.png" --output "lines.json"

# Compression
cd "2 Taylor"
python main.py --input-json "../symbols.json" --lines-json "../lines.json" --output-dir "../output"

# Connection finding
cd "3 Ryan"
python find_connections.py "ryan_input.json" --output-dir "../output"

# Visualization
cd "5 Koda"
python Visualizer.py --graph-json "graph.json" --output "graph_viz.png"
```

### Adding New Features

#### To modify line detection sensitivity:

Edit `1 Dalton/line_trace.py`:
```python
HOUGH_THRESHOLD = 30        # Lower = more lines detected
HOUGH_MIN_LINE_LENGTH = 15  # Lower = shorter lines kept
```

#### To change compression grid size:

Edit `pipeline.py` Stage 3:
```python
"--compression-index", "2"  # 0=32x32, 1=16x16, 2=8x8
```

#### To adjust symbol detection:

Edit `1 Dalton/run_new_best.py`:
```python
CONF = 0.20                    # Lower = more detections
KEEP_TOP_BREAKERS = 30         # Increase max breakers
```

---

## 🧪 Testing

### Run Test Suite

```bash
# Test with provided sample JSON (skips detection)
python pipeline.py --mode json test_bs_connected.json

# Test with sample image (full pipeline)
python pipeline.py "1 Dalton/bs.png"
```

### Expected Results

✅ **Pipeline completes successfully**
✅ **All 6 stages execute without errors**
✅ **13+ output files generated**
✅ **JSON files are valid and parseable**
✅ **Images display correctly**

### Verification Checklist

- [ ] `bs_connected.json` contains detected symbols
- [ ] `lines.json` contains detected lines
- [ ] `graph.json` contains connection graph
- [ ] `original_bboxes.png` shows green boxes and red lines
- [ ] `compressed_8x8.png` shows grid representation
- [ ] `output.png` shows labeled symbols
- [ ] `graph_visualized.png` shows matplotlib graph
- [ ] No Python exceptions or errors
- [ ] Pipeline completes in < 30 seconds

---

## 📊 Performance Notes

**Typical Runtime (on standard hardware):**
- Symbol Detection: 5-10 seconds
- Line Detection: 1-2 seconds
- Compression: 1-2 seconds
- Connection Finding: < 1 second
- Visualization: 1-2 seconds

**Total:** ~10-15 seconds for complete pipeline

**Memory Usage:** ~500MB-1GB depending on image size

---

## 👥 Credits

**Module Authors:**
- **Dalton** - Symbol detection (YOLO) & line tracing (Hough)
- **Taylor** - Image compression & grid representation
- **Ryan** - Connection finding (BFS algorithm)
- **Alden** - JSON building & metadata enhancement
- **Koda** - Graph visualization (matplotlib)

**Pipeline Integration:** November 2025

---

## 📄 License

[Add your license information here]

---

## 🔗 Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [OpenCV Hough Line Transform](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)
- [uv Package Manager](https://github.com/astral-sh/uv)

---

**Questions or Issues?** Open an issue on GitHub or contact the development team.
