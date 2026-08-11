"""I/O utilities for astroglial morphology analysis."""

from .file_detection import detect_input_file, is_suite2p_plane, InputFileInfo, InputFormat
from .metadata_loader import load_metadata, load_suite2p_metadata

__all__ = [
    "detect_input_file",
    "is_suite2p_plane",
    "InputFileInfo",
    "InputFormat",
    "load_metadata",
    "load_suite2p_metadata",
]
