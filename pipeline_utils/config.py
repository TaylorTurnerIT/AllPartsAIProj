"""Configuration constants for the pipeline."""

from pathlib import Path


class PipelineConfig:
    """Configuration for the diagram analysis pipeline."""

    # Project structure
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_BASE_DIR = PROJECT_ROOT / "6 Output Image"

    # Module directories
    MODULE_1_DIR = PROJECT_ROOT / "1 Dalton"
    MODULE_2_DIR = PROJECT_ROOT / "2 Taylor"
    MODULE_3_DIR = PROJECT_ROOT / "3 Ryan"
    MODULE_4_DIR = PROJECT_ROOT / "4 Alden"
    MODULE_5_DIR = PROJECT_ROOT / "5 Koda"

    # Module scripts
    BATCH_DETECT_SCRIPT = MODULE_1_DIR / "batch_detect.py"
    COMPRESSION_SCRIPT = MODULE_2_DIR / "main.py"
    FORMAT_ADAPTER_SCRIPT = MODULE_3_DIR / "format_adapter.py"
    CONNECTION_SCRIPT = MODULE_3_DIR / "find_connections.py"
    JSON_BUILDER_MODULE = MODULE_4_DIR / "json_builder_fixed.py"
    VISUALIZER_SCRIPT = MODULE_5_DIR / "Visualizer.py"

    # Model files
    MODEL_PATH = MODULE_1_DIR / "symbol_classifier.pth"
    CLASSES_PATH = MODULE_1_DIR / "classes.json"

    # Compression settings
    DEFAULT_COMPRESSION_SIZE = 16
    COMPRESSION_SIZES = [8, 16, 32]
    BBOX_MULTIPLIER = 1.0
    ENABLE_COLOR_QUANTIZATION = True
    COLOR_THRESHOLD = 500.0

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5

    # Output file names
    BS_CONNECTED_JSON = "bs_connected.json"
    COMPRESSION_RESULTS_JSON = "compression_results.json"
    RYAN_INPUT_JSON = "ryan_input.json"
    GRAPH_JSON = "graph.json"
    OUTPUT_PNG = "output.png"
    DIAGRAM_JSON = "diagram.json"
    GRAPH_VISUALIZED_PNG = "graph_visualized.png"
    PIPELINE_LOG = "pipeline.log"

    # Supported input file extensions
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    JSON_EXTENSION = ".json"

    @classmethod
    def create_output_dir(cls, timestamp=None):
        """Create a timestamped output directory."""
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        output_dir = cls.OUTPUT_BASE_DIR / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @classmethod
    def validate_project_structure(cls):
        """Validate that the project structure is correct."""
        errors = []

        # Check required directories exist
        required_dirs = [
            cls.MODULE_1_DIR,
            cls.MODULE_2_DIR,
            cls.MODULE_3_DIR,
            cls.MODULE_4_DIR,
            cls.MODULE_5_DIR,
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                errors.append(f"Missing directory: {dir_path}")

        return errors
