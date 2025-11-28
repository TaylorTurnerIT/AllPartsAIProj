"""Custom exceptions for the pipeline."""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class InputValidationError(PipelineError):
    """Raised when input validation fails."""
    pass


class StageExecutionError(PipelineError):
    """Raised when a pipeline stage fails to execute."""

    def __init__(self, stage_name, message, original_error=None):
        self.stage_name = stage_name
        self.original_error = original_error
        super().__init__(f"Stage '{stage_name}' failed: {message}")


class OutputValidationError(PipelineError):
    """Raised when output validation fails."""

    def __init__(self, file_path, message):
        self.file_path = file_path
        super().__init__(f"Output validation failed for '{file_path}': {message}")
