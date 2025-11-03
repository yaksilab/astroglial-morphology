from astroglial_morphology import do_registration, get_logger, setup_logging
from astroglial_morphology.binary_utils import create_projections

import tifffile
import os
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Astroglial Morphology pipeline")
    parser.add_argument(
        "data_path", help="Path to the directory containing the TIFF files"
    )
    parser.add_argument(
        "--reg-tif",
        action="store_true",
        help="Whether to save registered TIFF",
        default=False,
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    logger.info("Astroglial Morphology pipeline starting")
    data_path = args.data_path

    tif_files = [f for f in os.listdir(data_path) if f.endswith(".tif")]
    if not tif_files:
        raise FileNotFoundError(f"No .tif file found in directory: {data_path}")
    if len(tif_files) > 1:
        logger.warning(
            f"Multiple .tif files found in directory, using first one: {tif_files[0]}"
        )

    tiff_path = os.path.join(data_path, tif_files[0])
    with tifffile.TiffFile(tiff_path) as tif:
        nframes = len(tif.pages)
        if tif.pages:
            # Extract metadata from ImageJ metadata if available
            imagej_metadata = tif.imagej_metadata

            # Try to extract from 'info' field first (for multi-series LIF files)
            if imagej_metadata and "Info" in imagej_metadata:
                info_str = imagej_metadata["Info"]
                logger.info("Found ImageJ info metadata, parsing...")

                # Parse the info string to extract SizeC, SizeT, SizeZ for each series
                # Look for patterns like "SeriesName SizeC = 1"
                import re

                # Extract series names
                series_pattern = r"Series \d+ Name = (.+)"
                series_matches = re.findall(series_pattern, info_str)
                logger.info(f"Found {len(series_matches)} series: {series_matches}")

                # Use the first series for now (you can modify this to select a specific series)
                if series_matches:
                    first_series = series_matches[0].strip()
                    logger.info(f"Using first series: {first_series}")

                    # Extract SizeC, SizeT, SizeZ for this series
                    size_c_pattern = rf"{re.escape(first_series)} SizeC = (\d+)"
                    size_t_pattern = rf"{re.escape(first_series)} SizeT = (\d+)"
                    size_z_pattern = rf"{re.escape(first_series)} SizeZ = (\d+)"

                    size_c_match = re.search(size_c_pattern, info_str)
                    size_t_match = re.search(size_t_pattern, info_str)
                    size_z_match = re.search(size_z_pattern, info_str)

                    if size_c_match and size_t_match and size_z_match:
                        nchannels = int(size_c_match.group(1))
                        total_frames = int(size_t_match.group(1))
                        nplanes = int(size_z_match.group(1))

                        # Try to get frame interval from DimensionDescription
                        # Look for time dimension info
                        time_dim_pattern = (
                            r"DimensionDescription #3\|Length = ([\d.e+-]+)"
                        )
                        time_length_match = re.search(time_dim_pattern, info_str)

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
                                "Could not extract frame interval from metadata, using default: 1.0 s"
                            )

                        logger.info(
                            f"Extracted from info: channels={nchannels}, planes={nplanes}, frames={total_frames}, interval={finterval}"
                        )
                    else:
                        logger.warning(
                            "Could not parse Size parameters from info, falling back to standard metadata"
                        )
                        raise ValueError("Incomplete metadata in info field")
                else:
                    logger.warning(
                        "No series found in info metadata, falling back to standard metadata"
                    )
                    raise ValueError("No series in info field")

            # Fallback to standard ImageJ metadata
            elif (
                imagej_metadata
                and "channels" in imagej_metadata
                and "frames" in imagej_metadata
                and "finterval" in imagej_metadata
            ):
                nchannels = imagej_metadata["channels"]
                nplanes = imagej_metadata.get("slices", 1)
                finterval = imagej_metadata["finterval"]
                logger.info("Using standard ImageJ metadata fields")
            else:
                logger.warning("No ImageJ metadata found, using shape-based extraction")
                sample_page = tif.pages[0]
                shape = sample_page.shape
                # For multi-dimensional TIFFs, shape might be (planes, channels, height, width) or similar
                if len(shape) >= 3:
                    nplanes = shape[0] if len(shape) == 4 else 1
                    nchannels = (
                        shape[1]
                        if len(shape) == 4
                        else (shape[0] if len(shape) == 3 else 1)
                    )
                else:
                    nplanes = 1
                    nchannels = 1
                finterval = 1.0  # Default value if not found

    logger.info(
        f"TIFF metadata: {nframes} frames, {nplanes} planes, {nchannels} channels"
    )

    nplanes_int = int(nplanes)
    nchannels_int = int(nchannels)
    fs = 1.0 / finterval
    frames_per_channel_per_plane = nframes // (nplanes_int * nchannels_int)
    logger.info(f"Frames per channel per plane: {frames_per_channel_per_plane}")

    nimg_init = min(int(frames_per_channel_per_plane * 0.15), 300)
    batch_size = min(int(frames_per_channel_per_plane * 1), 500)
    logger.info(f"nimg_init: {nimg_init}, batch_size: {batch_size}")

    user_options = {
        "save_path0": "",
        "save_folder": [],
        "nplanes": nplanes,
        "nchannels": nchannels,
        "functional_chan": 1,
        "tau": 3,
        "fs": fs,
        "multiplane_parallel": False,
        "combined": True,
        "do_registration": True,
        "two_step_registration": False,
        "keep_movie_raw": False,
        "nimg_init": nimg_init,
        "batch_size": batch_size,
        "maxregshift": 0.11,
        "align_by_chan": 1,
        "subpixel": 10,
        "nonrigid": False,
        "roidetect": False,
        "spikedetect": False,
        "reg_tif": args.reg_tif,
    }

    logger.debug("User options: %s", user_options)
    do_registration(data_path, user_options)
    logger.info("Registration completed")
    logger.info("Creating projections")
    create_projections(data_path + "/suite2p", batch_size=batch_size)
    logger.info("Projections created")
    logger.info("Astroglial Morphology pipeline completed")


if __name__ == "__main__":
    main()
