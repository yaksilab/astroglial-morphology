"""Utilities for extracting metadata from TIFF files."""

import logging
import re
import tifffile

logger = logging.getLogger(__name__)


class TiffMetadata:
    """Container for TIFF metadata."""

    def __init__(
        self,
        nframes: int,
        nchannels: int,
        nplanes: int,
        finterval: float,
        pix_resolution: float,
    ):
        self.nframes = nframes
        self.nchannels = nchannels
        self.nplanes = nplanes
        self.finterval = finterval
        self.pix_resolution = pix_resolution

    @property
    def fs(self) -> float:
        """Sampling frequency (Hz)."""
        return 1.0 / self.finterval

    @property
    def frames_per_channel_per_plane(self) -> int:
        """Calculate frames per channel per plane."""
        return self.nframes // (self.nplanes * self.nchannels)

    def __repr__(self) -> str:
        return (
            f"TiffMetadata(nframes={self.nframes}, nchannels={self.nchannels}, "
            f"nplanes={self.nplanes}, finterval={self.finterval}, "
            f"pix_resolution={self.pix_resolution})"
        )


def extract_tiff_metadata(tiff_path: str) -> TiffMetadata:
    """
    Extract metadata from a TIFF file using tifffile's built-in metadata support.

    This function leverages tifffile's imagej_metadata property and parses the
    Info field when present (for multi-series LIF files).

    Args:
        tiff_path: Path to the TIFF file

    Returns:
        TiffMetadata object containing extracted metadata

    Raises:
        FileNotFoundError: If TIFF file doesn't exist
        ValueError: If required metadata is missing
    """
    with tifffile.TiffFile(tiff_path) as tif:
        nframes = len(tif.pages)

        if not tif.pages:
            raise ValueError("TIFF file has no pages")

        # Use tifffile's built-in imagej_metadata parsing
        imagej_meta = tif.imagej_metadata

        if not imagej_meta:
            raise ValueError(
                "No ImageJ metadata found in TIFF file. "
                "Proper ImageJ metadata is required to extract frame information, "
                "channels, planes, and temporal resolution."
            )

        if "Info" not in imagej_meta:
            raise ValueError(
                "ImageJ metadata found but 'Info' field is missing. "
                "The 'Info' field is required for extracting detailed metadata from LIF-derived TIFF files."
            )

        # Parse Info field for multi-series LIF files
        info_str = imagej_meta["Info"]
        logger.info("Found ImageJ Info metadata, parsing...")

        # Extract series names
        series_pattern = r"Series \d+ Name = (.+)"
        series_matches = re.findall(series_pattern, info_str)
        logger.info(f"Found {len(series_matches)} series: {series_matches}")

        if series_matches:
            # Use the first series
            first_series = series_matches[0].strip()
            logger.info(f"Using first series: {first_series}")

            # Extract SizeC, SizeT, SizeZ for this series
            # Try two patterns: with series name prefix and without
            size_c_pattern_with_prefix = rf"{re.escape(first_series)} SizeC = (\d+)"
            size_c_pattern_simple = r"SizeC = (\d+)"

            size_t_pattern_with_prefix = rf"{re.escape(first_series)} SizeT = (\d+)"
            size_t_pattern_simple = r"SizeT = (\d+)"

            size_z_pattern_with_prefix = rf"{re.escape(first_series)} SizeZ = (\d+)"
            size_z_pattern_simple = r"SizeZ = (\d+)"

            # Try with series name prefix first, then without
            size_c_match = re.search(size_c_pattern_with_prefix, info_str) or re.search(
                size_c_pattern_simple, info_str
            )
            size_t_match = re.search(size_t_pattern_with_prefix, info_str) or re.search(
                size_t_pattern_simple, info_str
            )
            size_z_match = re.search(size_z_pattern_with_prefix, info_str) or re.search(
                size_z_pattern_simple, info_str
            )

            # Extract what we can, use defaults for missing values
            nchannels = int(size_c_match.group(1)) if size_c_match else 1
            nplanes = int(size_z_match.group(1)) if size_z_match else 1

            # We need at least SizeT to proceed
            if size_t_match:

                # Try to get frame interval from DimensionDescription or finterval
                # First check if finterval is directly available in imagej_meta
                finterval = imagej_meta.get("finterval")

                if not finterval:
                    # Try to get from CycleTime field (frame time in seconds)
                    cycle_time_pattern = r"CycleTime = ([\d.e+-]+)"
                    cycle_time_match = re.search(cycle_time_pattern, info_str)

                    if cycle_time_match:
                        finterval = float(cycle_time_match.group(1))
                        logger.info(
                            f"Extracted frame interval from CycleTime: {finterval} s"
                        )
                    else:
                        # Try to calculate from DimensionDescription with time dimension (DimID = 4)
                        total_frames = int(size_t_match.group(1))
                        # Look for time dimension by DimID = 4, not by a specific dimension number
                        time_dim_pattern = r"DimensionDescription #\d+\|.*?DimID = 4.*?\|Length = ([\d.e+-]+)"
                        time_length_match = re.search(
                            time_dim_pattern, info_str, re.DOTALL
                        )

                        if time_length_match:
                            total_time = float(time_length_match.group(1))
                            finterval = (
                                total_time / total_frames if total_frames > 0 else 1.0
                            )
                            logger.info(
                                f"Calculated frame interval from total time: {finterval} s"
                            )
                        else:
                            finterval = 1.0
                            logger.warning(
                                "Could not extract frame interval, using default: 1.0 s"
                            )
                else:
                    logger.info(f"Using finterval from ImageJ metadata: {finterval} s")

                # Extract pixel resolution from X dimension (DimID = 1)
                # Length is in meters, NumberOfElements is pixel count
                pix_resolution = 1.0  # Default
                # Search for X dimension by DimID = 1, not by a specific dimension number
                dim_x_length_pattern = (
                    r"DimensionDescription #\d+\|.*?DimID = 1.*?\|Length = ([\d.e+-]+)"
                )
                dim_x_elements_pattern = r"DimensionDescription #\d+\|.*?DimID = 1.*?\|NumberOfElements = (\d+)"
                dim_x_unit_pattern = (
                    r"DimensionDescription #\d+\|.*?DimID = 1.*?\|Unit = (\w+)"
                )

                length_match = re.search(dim_x_length_pattern, info_str, re.DOTALL)
                elements_match = re.search(dim_x_elements_pattern, info_str, re.DOTALL)
                unit_match = re.search(dim_x_unit_pattern, info_str, re.DOTALL)

                if length_match and elements_match:
                    length_value = float(length_match.group(1))
                    num_pixels = int(elements_match.group(1))
                    unit = unit_match.group(1) if unit_match else "m"

                    # Convert to micrometers based on unit
                    if unit == "m":
                        length_um = length_value * 1e6
                    elif unit == "mm":
                        length_um = length_value * 1e3
                    elif unit in ["um", "µm", "micrometer"]:
                        length_um = length_value
                    else:
                        logger.warning(f"Unknown unit '{unit}', assuming meters")
                        length_um = length_value * 1e6

                    pix_resolution = num_pixels / length_um
                    logger.info(
                        f"Calculated pixel resolution: {pix_resolution:.3f} pixels/micrometer "
                        f"({num_pixels} pixels / {length_um:.3f} µm)"
                    )
                else:
                    logger.warning(
                        "Could not extract pixel resolution, using default: 1.0"
                    )

                # Validate extracted dimensions match total frames
                expected_frames = nchannels * nplanes * int(size_t_match.group(1))
                if expected_frames != nframes:
                    # Check if the mismatch is due to single-channel extraction
                    if nchannels > 1 and expected_frames == nframes * nchannels:
                        logger.warning(
                            f"Frame count mismatch: expected {expected_frames} "
                            f"(C={nchannels} × Z={nplanes} × T={size_t_match.group(1)}), "
                            f"but file has {nframes} pages. "
                            f"Detected single-channel extraction - adjusting channels from {nchannels} to 1."
                        )
                        nchannels = 1
                    else:
                        logger.warning(
                            f"Frame count mismatch: expected {expected_frames} "
                            f"(C={nchannels} × Z={nplanes} × T={size_t_match.group(1)}), "
                            f"but file has {nframes} pages. Using file page count."
                        )

                logger.info(
                    f"Extracted from Info: channels={nchannels}, planes={nplanes}, "
                    f"finterval={finterval}, pix_resolution={pix_resolution:.3f}"
                )
                return TiffMetadata(
                    nframes, nchannels, nplanes, finterval, pix_resolution
                )
            else:
                raise ValueError(
                    "Could not extract SizeT from Info field. "
                    "Proper ImageJ metadata with time series information is required."
                )

        raise ValueError(
            "Could not parse Info field or series information. "
            "Proper ImageJ metadata is required."
        )
