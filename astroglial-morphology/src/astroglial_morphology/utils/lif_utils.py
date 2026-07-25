"""Utilities for extracting metadata from LIF files and converting to Suite2p binary format."""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Union
from readlif.reader import LifFile

# Import Metadata class to maintain compatibility with existing pipeline
from .tiff_utils import Metadata

logger = logging.getLogger(__name__)


def extract_lif_metadata(lif_path: str, series_index: int = 0) -> Metadata:
    """
    Extract metadata from a LIF file using readlif.

    Args:
        lif_path: Path to the LIF file
        series_index: Index of the series to extract (default: 0, first series)

    Returns:
        Metadata object containing extracted metadata

    Raises:
        FileNotFoundError: If LIF file doesn't exist
        ValueError: If required metadata is missing or series index is invalid
    """
    lif_path_obj = Path(lif_path)
    if not lif_path_obj.exists():
        raise FileNotFoundError(f"LIF file not found: {lif_path}")

    lif = LifFile(lif_path)

    if not lif.image_list:
        raise ValueError(f"No images found in LIF file: {lif_path}")

    if series_index >= len(lif.image_list):
        raise ValueError(
            f"Series index {series_index} out of range. "
            f"LIF file contains {len(lif.image_list)} series."
        )

    # Warn if multiple series detected
    if len(lif.image_list) > 1:
        logger.warning(
            f"Multiple series detected ({len(lif.image_list)} series). "
            f"Only series {series_index} ('{lif.image_list[series_index]['name']}') will be processed."
        )

    # Get the first (or specified) image series
    img = lif.get_image(series_index)

    logger.info(f"Loading LIF series: {img.name}")
    logger.info(f"Dimensions: {img.dims}")
    logger.info(f"Channels: {img.channels}")

    # Warn if multiple channels detected
    if img.channels > 1:
        logger.warning(
            f"Multiple channels detected ({img.channels} channels). "
            f"Up to two channels can be converted for Suite2p processing."
        )

    # Extract dimensions
    # img.dims is a namedtuple with attributes: x, y, z, t, m
    # where 't' is time (frames), 'x' is width, 'y' is height, 'z' is z-slices
    dims = img.dims

    frames_per_channel_per_plane = dims.t
    nchannels = img.channels
    nplanes = dims.z  # Number of z-planes
    nframes = frames_per_channel_per_plane * nchannels * nplanes

    # Extract pixel resolution (microns per pixel)
    # img.scale is a tuple of (x_scale, y_scale, z_scale, t_scale)
    # Documentation says it's in meters, but testing shows it's actually in microns
    if len(img.scale) >= 2:
        # Scale appears to already be in microns/pixel based on typical imaging values
        pix_resolution = img.scale[0]  # x-scale in microns/pixel
        logger.info(f"Pixel resolution: {pix_resolution:.4f} pixels/micron")
    else:
        logger.warning("Could not extract pixel resolution from LIF, using default 1.0")
        pix_resolution = 1.0

    # Extract frame interval (time between frames in seconds)
    # img.scale[3] is time interval in seconds (if available)
    if len(img.scale) >= 4 and img.scale[3] > 0:
        finterval = img.scale[3]
        logger.info(f"Frame interval: {finterval:.4f} seconds")
    else:
        logger.warning("Could not extract frame interval from LIF, using default 1.0s")
        finterval = 1.0

    metadata = Metadata(
        nframes=nframes,
        nchannels=nchannels,
        nplanes=nplanes,
        finterval=finterval,
        pix_resolution=pix_resolution,
        series_name=img.name,
        series_index=series_index,
    )

    logger.info(f"LIF metadata: {metadata}")

    return metadata


def _normalize_channel_indices(
    channel_index: Optional[Union[int, Sequence[int]]],
    available_channels: int,
) -> list[int]:
    if channel_index is None:
        if available_channels > 2:
            logger.warning(
                "LIF file has %d channels; only channels 0 and 1 are supported",
                available_channels,
            )
        return list(range(min(available_channels, 2)))

    if isinstance(channel_index, int):
        channel_indices = [channel_index]
    else:
        channel_indices = list(channel_index)

    if not channel_indices:
        raise ValueError("At least one channel must be selected for conversion")
    if len(channel_indices) > 2:
        raise ValueError("Suite2p conversion supports at most two channels")
    if len(set(channel_indices)) != len(channel_indices):
        raise ValueError("Duplicate channel indices are not allowed")

    for index in channel_indices:
        if index < 0 or index >= available_channels:
            raise ValueError(
                f"Channel index {index} out of range. "
                f"LIF file has {available_channels} channels."
            )

    return channel_indices


def _frame_to_int16(frame_np: np.ndarray) -> np.ndarray:
    if frame_np.dtype == np.uint16:
        return (frame_np // 2).astype(np.int16)
    if frame_np.dtype == np.uint8:
        return frame_np.astype(np.int16) * 128
    return frame_np.astype(np.int16)


def lif_to_suite2p_binary(
    lif_path: str,
    output_dir: str,
    series_index: int = 0,
    channel_index: Optional[Union[int, Sequence[int]]] = None,
    plane_index: int = 0,
) -> Dict[str, Any]:
    """
    Convert LIF file to Suite2p binary format.

    Creates a Suite2p-compatible directory structure with:
    - suite2p/plane{plane_index}/data.bin (binary frame data)
    - suite2p/plane{plane_index}/ops.npy (metadata)

    Args:
        lif_path: Path to the LIF file
        output_dir: Directory where suite2p folder will be created
        series_index: Index of the series to convert (default: 0)
        channel_index: Index of the channel to extract (default: 0)
        plane_index: Index of the z-plane to extract (default: 0)

    Returns:
        Dictionary containing Suite2p ops metadata

    Raises:
        FileNotFoundError: If LIF file doesn't exist
        ValueError: If indices are out of range
    """
    logger.info(f"Converting LIF file to Suite2p binary format: {lif_path}")

    # Extract metadata
    metadata = extract_lif_metadata(lif_path, series_index)

    channel_indices = _normalize_channel_indices(channel_index, metadata.nchannels)

    if plane_index >= metadata.nplanes:
        raise ValueError(
            f"Plane index {plane_index} out of range. "
            f"LIF file has {metadata.nplanes} planes."
        )

    # Create output directory structure
    output_path = Path(output_dir)
    suite2p_path = output_path / "suite2p"
    plane_path = suite2p_path / f"plane{plane_index}"
    plane_path.mkdir(parents=True, exist_ok=True)

    bin_file_paths = [
        plane_path / ("data.bin" if output_index == 0 else "data_chan2.bin")
        for output_index, _ in enumerate(channel_indices)
    ]
    ops_file_path = plane_path / "ops.npy"

    logger.info(
        "Creating Suite2p binary files for channels %s: %s",
        channel_indices,
        [str(path) for path in bin_file_paths],
    )

    # Open LIF file and get image series
    lif = LifFile(lif_path)
    img = lif.get_image(series_index)

    # Get frame dimensions
    dims = img.dims
    Ly = dims.y  # Frame height
    Lx = dims.x  # Frame width
    nframes = dims.t  # Number of time frames

    logger.info(f"Frame dimensions: {Ly} x {Lx}, {nframes} frames")

    frame_counts = {channel: 0 for channel in channel_indices}
    with open(bin_file_paths[0], "wb") as f0:
        file_handles = [f0]
        if len(bin_file_paths) == 2:
            file_handles.append(open(bin_file_paths[1], "wb"))

        try:
            for t_idx in range(nframes):
                for output_index, source_channel in enumerate(channel_indices):
                    frame_pil = img.get_frame(
                        z=plane_index, t=t_idx, c=source_channel
                    )
                    frame_np = np.array(frame_pil)
                    frame_data = _frame_to_int16(frame_np)

                    if frame_data.shape != (Ly, Lx):
                        logger.warning(
                            f"Frame {t_idx} channel {source_channel} has unexpected "
                            f"shape {frame_data.shape}, expected ({Ly}, {Lx})"
                        )

                    file_handles[output_index].write(frame_data.tobytes())
                    frame_counts[source_channel] += 1

                if (t_idx + 1) % 100 == 0 or t_idx == nframes - 1:
                    logger.info(f"Processed {t_idx + 1}/{nframes} frames")
        finally:
            for handle in file_handles[1:]:
                handle.close()

    for source_channel, path in zip(channel_indices, bin_file_paths):
        logger.info(
            "Successfully wrote %d frames for channel %d to %s",
            frame_counts[source_channel],
            source_channel,
            path,
        )

    # Create ops dictionary (Suite2p metadata)
    # When input_format='binary', Suite2p expects Lys and Lxs arrays
    ops = {
        "Ly": int(Ly),
        "Lx": int(Lx),
        "Lys": [int(Ly)],  # Array of frame heights for each plane
        "Lxs": [int(Lx)],  # Array of frame widths for each plane
        "nframes": int(nframes),
        "nchannels": int(len(channel_indices)),
        "source_nchannels": int(metadata.nchannels),
        "channel_indices": channel_indices,
        "nplanes": int(metadata.nplanes),
        "fs": float(metadata.fs),
        "tau": 3.0,  # Default decay time constant
        "reg_file": str(bin_file_paths[0]),
        "functional_chan": 1,
        "meanImg": None,  # Will be calculated later by Suite2p or BinaryDataProcessor
        # Add other Suite2p-specific parameters with defaults
        "do_registration": True,
        "two_step_registration": False,
        "keep_movie_raw": False,
        "smooth_sigma": 1.15,
        "maxregshift": 0.11,
        "align_by_chan": 1,
        "subpixel": 10,
        "nonrigid": False,
    }
    if len(bin_file_paths) > 1:
        ops["reg_file_chan2"] = str(bin_file_paths[1])
        ops["meanImg_chan2"] = None

    # Save ops file (allow_pickle=True required for dict)
    np.save(ops_file_path, ops, allow_pickle=True)  # type: ignore
    logger.info(f"Saved ops metadata to {ops_file_path}")

    return ops
