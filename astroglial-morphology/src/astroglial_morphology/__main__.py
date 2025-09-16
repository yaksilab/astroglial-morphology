from astroglial_morphology import do_registration, get_logger, setup_logging


def main() -> None:
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Astroglial Morphology pipeline starting")

    data_path = (
        r"C:\Users\javid.rezai\YaksiLab\duygu\data\glast-gfp_OT_baseline_C=2_xyt"
    )

    user_options = {
        "save_path0": "",
        "save_folder": [],
        "nplanes": 1,
        "nchannels": 2,
        "functional_chan": 1,
        "tau": 3,
        "fs": 0.1942,
        "multiplane_parallel": False,
        "combined": True,
        "do_registration": True,
        "two_step_registration": False,
        "keep_movie_raw": False,
        "nimg_init": 100,
        "batch_size": 300,
        "maxregshift": 0.11,
        "align_by_chan": 1,
        "subpixel": 10,
        "nonrigid": False,
        "roidetect": False,
        "spikedetect": False,
    }

    logger.debug("User options: %s", user_options)
    do_registration(data_path, user_options)
    logger.info("Astroglial Morphology pipeline completed")


if __name__ == "__main__":
    main()
