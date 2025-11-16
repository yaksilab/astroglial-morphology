"""I/O utilities for astroglial morphology analysis."""

from .file_detection import detect_input_file, InputFileInfo, InputFormat
from .metadata_loader import load_metadata

__all__ = [
    "detect_input_file",
    "InputFileInfo",
    "InputFormat",
    "load_metadata",
]
