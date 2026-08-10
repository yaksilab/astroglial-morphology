"""Metadata loading utilities for different file formats."""

import logging
import math
from pathlib import Path
from typing import Any, Optional
import numpy as np
from .file_detection import InputFileInfo, InputFormat
from ..utils.tiff_utils import extract_tiff_metadata
from ..utils.lif_utils import extract_lif_metadata
from ..utils.tiff_utils import Metadata

logger = logging.getLogger(__name__)


def load_suite2p_metadata(
    plane_path: str | Path,
    pixels_per_micron: Optional[float] = None,
) -> Metadata:
    """Build the pipeline metadata needed for an existing Suite2p plane."""

    plane = Path(plane_path)
    ops_path = plane / "ops.npy"
    if not ops_path.is_file() or not (plane / "data.bin").is_file():
        raise FileNotFoundError(
            f"Suite2p plane must contain ops.npy and data.bin: {plane}"
        )
    try:
        ops = np.load(ops_path, allow_pickle=True).item()
    except Exception as exc:
        raise ValueError(f"Could not read Suite2p ops.npy: {ops_path}") from exc

    required = ("Ly", "Lx", "nframes", "fs")
    missing = [name for name in required if name not in ops]
    if missing:
        raise ValueError(f"Suite2p ops.npy is missing required fields: {', '.join(missing)}")
    try:
        ly, lx, frames = int(ops["Ly"]), int(ops["Lx"]), int(ops["nframes"])
        fs = float(ops["fs"])
        channels = int(ops.get("nchannels", 1))
        planes = int(ops.get("nplanes", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Suite2p ops.npy contains invalid dimensions or frame rate") from exc
    if ly <= 0 or lx <= 0 or frames <= 0 or channels not in {1, 2} or planes <= 0:
        raise ValueError("Suite2p ops.npy must have positive dimensions, frames, and one or two channels")
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("Suite2p ops.npy must contain a positive finite fs")
    expected_bytes = ly * lx * frames * np.dtype(np.int16).itemsize
    binary_paths = [plane / "data.bin"]
    if channels == 2:
        channel_two = plane / "data_chan2.bin"
        if not channel_two.is_file():
            raise FileNotFoundError(
                f"Suite2p ops.npy declares two channels but data_chan2.bin is missing: {plane}"
            )
        binary_paths.append(channel_two)
    for binary_path in binary_paths:
        if binary_path.stat().st_size < expected_bytes:
            raise ValueError(
                f"Suite2p binary is shorter than ops.npy declares ({expected_bytes} bytes): "
                f"{binary_path}"
            )

    # Metadata historically requires a pixel-resolution value.  A missing
    # calibration is represented by Pipeline.pixels_per_micron=None; this
    # placeholder is only used by legacy non-physical single-model operations.
    resolution = 1.0 if pixels_per_micron is None else float(pixels_per_micron)
    return Metadata(
        nframes=frames * channels * planes,
        nchannels=channels,
        nplanes=planes,
        finterval=1.0 / fs,
        pix_resolution=resolution,
    )


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
    elif file_info.format == InputFormat.SUITE2P:
        metadata = load_suite2p_metadata(file_info.path)
    else:
        raise ValueError(f"Unsupported file format: {file_info.format}")

    logger.info(
        f"Frames per channel per plane: {metadata.frames_per_channel_per_plane}"
    )

    return metadata
