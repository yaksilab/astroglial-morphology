from astroglial_morphology import do_registration, get_logger, setup_logging
from astroglial_morphology.binary_utils import create_projections
import numpy as np
from utils.tiff_utils import extract_tiff_metadata
from astroglial_morphology.segmentation import Segmentation
from astroglial_morphology.classifier import classify_cells
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

    segmenter = Segmentation(
        model_path=r"C:\Users\javid.rezai\YaksiLab\duygu\astroglial-morphology\astroglial-morphology\src\models\CP3_S4_1_0001_3000",
        gpu=False,
    )

    data_path = args.data_path

    tif_files = [f for f in os.listdir(data_path) if f.endswith(".tif")]
    if not tif_files:
        raise FileNotFoundError(f"No .tif file found in directory: {data_path}")
    if len(tif_files) > 1:
        logger.warning(
            f"Multiple .tif files found in directory, using first one: {tif_files[0]}"
        )

    tiff_path = os.path.join(data_path, tif_files[0])

    # Extract TIFF metadata using utility
    metadata = extract_tiff_metadata(tiff_path)
    logger.info(
        f"TIFF metadata: {metadata.nframes} frames, {metadata.nplanes} planes, "
        f"{metadata.nchannels} channels, {metadata.finterval}s interval"
    )
    logger.info(
        f"Frames per channel per plane: {metadata.frames_per_channel_per_plane}"
    )

    nimg_init = min(int(metadata.frames_per_channel_per_plane * 0.15), 300)
    batch_size = min(int(metadata.frames_per_channel_per_plane * 1), 500)
    logger.info(f"nimg_init: {nimg_init}, batch_size: {batch_size}")

    user_options = {
        "save_path0": "",
        "save_folder": [],
        "nplanes": metadata.nplanes,
        "nchannels": metadata.nchannels,
        "functional_chan": 1,
        "tau": 3,
        "fs": metadata.fs,
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
    projections = create_projections(data_path + "/suite2p", batch_size=batch_size)
    logger.info("Projections created")
    logger.info("Doing segmentation on mean projection")
    save_path = os.path.join(data_path, "suite2p", "plane0", "mean_image")
    diameter = (
        metadata.pix_resolution * 31.35
    )  # approximate average astrocyte diameter in microns
    masks = segmenter.segment_img(
        projections["mean"], save_path, diameter=diameter + 10
    )
    labels = np.unique(masks)
    logger.info(f"Masks found with labels: {len(labels)}")
    logger.info(
        f"Please inspect the segmentation masks saved and make manual corrections if needed. Make manual correction by opening the segmented image file in cellpose"
    )
    manual_correc = (
        True
        if input("Did you make manual corrections to the masks y/n?") == "y"
        else False
    )

    if manual_correc:
        logger.info("loding manually corrected masks")
        masks = np.load(
            os.path.join(data_path, "suite2p", "plane0", "mean_image_seg.npy"),
            allow_pickle=True,
        ).item()["masks"]

    logger.info("Doing classification of astrocyte morphology")
    classification = classify_cells(masks=masks, neck_distance=int(diameter * 0.47))

    logger.info("Classifications: ", classification)
    logger.info("Classification completed")
    logger.info("Astroglial Morphology pipeline completed")


if __name__ == "__main__":
    main()
