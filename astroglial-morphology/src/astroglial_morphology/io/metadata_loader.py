"""Metadata loading utilities for different file formats."""

import logging
from typing import Any
from .file_detection import InputFileInfo, InputFormat
from utils.tiff_utils import extract_tiff_metadata
from utils.lif_utils import extract_lif_metadata

logger = logging.getLogger(__name__)


def load_metadata(file_info: InputFileInfo) -> Any:
    """
    Load metadata from input file.

    Provides a unified interface for loading metadata from different
    file formats (TIFF, LIF, etc.).

    Args:
        file_info: InputFileInfo object with file path and format

    Returns:
        TiffMetadata object (unified metadata structure)

    Raises:
        ValueError: If file format is not supported
    """
    if file_info.format == InputFormat.LIF:
        metadata = extract_lif_metadata(file_info.path_str, series_index=0)
        logger.info(
            f"LIF metadata: {metadata.nframes} frames, {metadata.nplanes} planes, "
            f"{metadata.nchannels} channels, {metadata.finterval}s interval"
        )
    elif file_info.format == InputFormat.TIFF:
        metadata = extract_tiff_metadata(file_info.path_str)
        logger.info(
            f"TIFF metadata: {metadata.nframes} frames, {metadata.nplanes} planes, "
            f"{metadata.nchannels} channels, {metadata.finterval}s interval"
        )
    else:
        raise ValueError(f"Unsupported file format: {file_info.format}")

    logger.info(
        f"Frames per channel per plane: {metadata.frames_per_channel_per_plane}"
    )

    return metadata
