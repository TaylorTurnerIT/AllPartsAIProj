"""Input and output validation for the pipeline."""

import json
from pathlib import Path
from .errors import InputValidationError, OutputValidationError
from .config import PipelineConfig


class InputValidator:
    """Validates pipeline inputs."""

    @staticmethod
    def validate_input_file(file_path):
        """
        Validate that the input file exists and is a supported type.

        Args:
            file_path: Path to the input file

        Returns:
            tuple: (Path object, input_type) where input_type is 'image' or 'json'

        Raises:
            InputValidationError: If validation fails
        """
        path = Path(file_path)

        if not path.exists():
            raise InputValidationError(f"Input file does not exist: {file_path}")

        if not path.is_file():
            raise InputValidationError(f"Input path is not a file: {file_path}")

        suffix = path.suffix.lower()

        if suffix in PipelineConfig.IMAGE_EXTENSIONS:
            return path, 'image'
        elif suffix == PipelineConfig.JSON_EXTENSION:
            return path, 'json'
        else:
            supported = ", ".join(
                list(PipelineConfig.IMAGE_EXTENSIONS) + [PipelineConfig.JSON_EXTENSION]
            )
            raise InputValidationError(
                f"Unsupported file type: {suffix}. Supported types: {supported}"
            )

    @staticmethod
    def validate_bs_connected_json(file_path):
        """
        Validate bs_connected.json structure.

        Args:
            file_path: Path to the JSON file

        Raises:
            InputValidationError: If validation fails
        """
        try:
            with open(file_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise InputValidationError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise InputValidationError(f"Failed to read {file_path}: {e}")

        if "symbols" not in data:
            raise InputValidationError(
                f"{file_path} missing required 'symbols' field"
            )

        if not isinstance(data["symbols"], list):
            raise InputValidationError(
                f"{file_path} 'symbols' field must be a list"
            )

        # Validate each symbol has required fields
        required_fields = ["id", "name", "bbox", "center"]
        for i, symbol in enumerate(data["symbols"]):
            for field in required_fields:
                if field not in symbol:
                    raise InputValidationError(
                        f"{file_path} symbol {i} missing required field '{field}'"
                    )


class OutputValidator:
    """Validates pipeline outputs."""

    @staticmethod
    def validate_file_exists(file_path, file_description="File"):
        """
        Validate that an output file was created.

        Args:
            file_path: Path to check
            file_description: Description for error messages

        Raises:
            OutputValidationError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise OutputValidationError(
                path, f"{file_description} was not created"
            )
        if path.stat().st_size == 0:
            raise OutputValidationError(
                path, f"{file_description} is empty"
            )

    @staticmethod
    def validate_json_file(file_path, required_fields=None):
        """
        Validate that a JSON file is valid and contains required fields.

        Args:
            file_path: Path to JSON file
            required_fields: List of required top-level fields

        Raises:
            OutputValidationError: If validation fails
        """
        path = Path(file_path)

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise OutputValidationError(path, f"Invalid JSON: {e}")
        except Exception as e:
            raise OutputValidationError(path, f"Failed to read file: {e}")

        if required_fields:
            missing = [field for field in required_fields if field not in data]
            if missing:
                raise OutputValidationError(
                    path,
                    f"Missing required fields: {', '.join(missing)}"
                )

    @staticmethod
    def validate_image_file(file_path):
        """
        Validate that an image file exists and is readable.

        Args:
            file_path: Path to image file

        Raises:
            OutputValidationError: If validation fails
        """
        path = Path(file_path)

        if not path.exists():
            raise OutputValidationError(path, "Image file was not created")

        # Try to open with PIL to verify it's a valid image
        try:
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            raise OutputValidationError(path, f"Invalid image file: {e}")
