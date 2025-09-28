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
            if (
                imagej_metadata
                and "channels" in imagej_metadata
                and "frames" in imagej_metadata
            ):
                nchannels = imagej_metadata["channels"]
                nplanes = imagej_metadata.get(
                    "slices", 1
                )  # 'slices' for planes in ImageJ
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

    logger.info(
        f"TIFF metadata: {nframes} frames, {nplanes} planes, {nchannels} channels"
    )

    nplanes_int = int(nplanes)
    nchannels_int = int(nchannels)
    frames_per_channel_per_plane = nframes // (nplanes_int * nchannels_int)
    logger.info(f"Frames per channel per plane: {frames_per_channel_per_plane}")

    nimg_init = min(int(frames_per_channel_per_plane * 0.15), 300)
    batch_size = min(int(frames_per_channel_per_plane * 0.2), 500)
    logger.info(f"nimg_init: {nimg_init}, batch_size: {batch_size}")

    user_options = {
        "save_path0": "",
        "save_folder": [],
        "nplanes": nplanes,
        "nchannels": nchannels,
        "functional_chan": 1,
        "tau": 3,
        "fs": 0.1942,
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
    }

    logger.debug("User options: %s", user_options)
    do_registration(data_path, user_options)
    logger.info("Registration completed")
    logger.info("Creating projections")
    create_projections(data_path + "/suite2p")
    logger.info("Projections created")
    logger.info("Astroglial Morphology pipeline completed")


if __name__ == "__main__":
    main()
