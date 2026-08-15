"""
Main entry point for the astroglial morphology pipeline.

Usage:
    python -m astroglial_morphology <data_path> [--reg-tif] [--model PATH] [--gpu]
"""

import argparse
import sys
from pathlib import Path

from astroglial_morphology import setup_logging, get_logger
from astroglial_morphology.correspondence import (
    VALID_SUBSEGMENTATION_MODES,
    SUBSEGMENTATION_MODE_EQUAL_LENGTH,
)
from astroglial_morphology.ensemble import (
    DEFAULT_PROFILE_NAME,
    ModelAssetResolver,
    load_ensemble_profile,
)
from astroglial_morphology.pipeline import Pipeline


def main() -> None:
    """Run the astroglial morphology analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Astroglial morphology analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process TIFF or LIF file (auto-detects if registration already done)
  python -m astroglial_morphology /path/to/data
  
  # Force re-registration even if already complete
  python -m astroglial_morphology /path/to/data --force-registration
  
  # Save registered TIFF (large files!)
  python -m astroglial_morphology /path/to/data --reg-tif
  
  # Use custom model
  python -m astroglial_morphology /path/to/data --model /path/to/model
  
  # Use GPU for segmentation
  python -m astroglial_morphology /path/to/data --gpu
        """,
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        help="Path to raw LIF/TIFF data or an existing Suite2p plane0 directory",
    )
    parser.add_argument(
        "--reg-tif",
        action="store_true",
        help="Save registered TIFF files (warning: creates large files)",
        default=False,
    )
    parser.add_argument(
        "--model",
        help="Path to custom Cellpose model (optional)",
        default=None,
    )
    parser.add_argument(
        "--segmentation-mode",
        choices=["single", "ensemble"],
        default="single",
        help="Segmentation strategy; single preserves the existing pipeline default",
    )
    parser.add_argument(
        "--ensemble-profile",
        default=DEFAULT_PROFILE_NAME,
        help="Packaged ensemble profile to use",
    )
    parser.add_argument(
        "--ensemble-config",
        default=None,
        help="Path to a complete custom three-role ensemble JSON profile",
    )
    parser.add_argument(
        "--pixels-per-micron",
        type=float,
        default=None,
        help="Physical calibration override, in pixels per micron",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Directory used to cache verified ensemble model assets",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        default=False,
        help="Download and verify ensemble models, then exit without processing data",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for segmentation",
        default=False,
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Unconditionally skip registration step (deprecated - auto-detected now)",
        default=False,
    )
    parser.add_argument(
        "--force-registration",
        action="store_true",
        help="Force registration even if already complete",
        default=False,
    )
    parser.add_argument(
        "--alignment-only",
        action="store_true",
        help="Stop after registration and projection creation",
        default=False,
    )
    parser.add_argument(
        "--registration-channel",
        type=int,
        choices=[0, 1],
        default=0,
        help="Zero-based channel used to calculate registration shifts",
    )
    parser.add_argument(
        "--regmetrics",
        action="store_true",
        help="Enable Suite2p registration metrics (can require substantial memory)",
        default=False,
    )
    parser.add_argument(
        "--manual-correction",
        action="store_true",
        help="Pause after automatic segmentation so masks can be corrected in Cellpose",
        default=False,
    )
    parser.add_argument(
        "--export-correspondence",
        action=argparse.BooleanOptionalAction,
        help=(
            "Generate correspondence matrix, subsegmented masks, and trace exports "
            "(default; use --no-export-correspondence to skip)"
        ),
        default=True,
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=5,
        help="Segment length of subsegment (in microns) used when exporting correspondence data",
    )
    parser.add_argument(
        "--correspondence-delta-x",
        type=float,
        default=20.0,
        help="Maximum x-distance for grouping cells while aligning correspondence",
    )
    parser.add_argument(
        "--subsegmentation-mode",
        choices=sorted(VALID_SUBSEGMENTATION_MODES),
        default=SUBSEGMENTATION_MODE_EQUAL_LENGTH,
        help="How to subsegment each cell: equal_length (fixed pixels) or compartments (soma/middle/distal)",
    )
    parser.add_argument(
        "--segmentation-image",
        choices=["mean", "max_projection"],
        default="mean",
        help="Projection image to segment on: mean or max_projection",
    )
    parser.add_argument(
        "--segmentation-channel",
        choices=["auto", "both", "0", "1"],
        default="auto",
        help="Channel mode to segment: auto (1 for two-channel data), both, 0, or 1",
    )
    parser.add_argument(
        "--trace-channels",
        default=None,
        help=(
            "Comma-separated zero-based channels to export traces for, e.g. 0 or 0,1 "
            "(the first selected channel is the Suite2p GUI primary trace)"
        ),
    )

    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    if args.download_models:
        try:
            profile, assets = load_ensemble_profile(
                profile_name=args.ensemble_profile,
                config_path=args.ensemble_config,
            )
            downloaded = ModelAssetResolver(assets, args.model_cache_dir).prefetch(profile)
            for role, path in downloaded.items():
                logger.info("Verified %s model: %s", role, path)
            return
        except Exception as e:
            logger.exception("Model prefetch failed: %s", e)
            sys.exit(1)

    if args.data_path is None:
        parser.error("data_path is required unless --download-models is used")
    if args.segmentation_mode == "ensemble" and args.model is not None:
        parser.error("--model applies to single-model mode; use --ensemble-config for ensemble models")

    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        sys.exit(1)

    if not data_path.is_dir():
        logger.error(f"Data path is not a directory: {data_path}")
        sys.exit(1)

    try:
        pipeline = Pipeline(
            data_path=str(data_path),
            model_path=args.model,
            use_gpu=args.gpu,
            reg_tif=args.reg_tif,
            segmentation_mode=args.segmentation_mode,
            ensemble_profile=args.ensemble_profile,
            ensemble_config=args.ensemble_config,
            pixels_per_micron=args.pixels_per_micron,
            model_cache_dir=args.model_cache_dir,
        )

        results = pipeline.run(
            skip_registration=args.skip_registration,
            force_registration=args.force_registration,
            manual_correction=args.manual_correction,
            export_correspondence=args.export_correspondence,
            correspondence_segment_length=args.segment_length,
            correspondence_delta_x=args.correspondence_delta_x,
            correspondence_subsegmentation_mode=args.subsegmentation_mode,
            segmentation_projection=args.segmentation_image,
            segmentation_channel=args.segmentation_channel,
            registration_channel=args.registration_channel,
            trace_channels=args.trace_channels,
            do_regmetrics=args.regmetrics,
            alignment_only=args.alignment_only,
        )

        logger.info("Pipeline completed successfully")
        logger.info(f"Results: {results['classification']}")
        if args.export_correspondence and results.get("correspondence"):
            corr_outputs = results["correspondence"]
            logger.info(
                "Correspondence matrix: %s (npy), %s (mat)",
                corr_outputs["correspondence_matrix_path"],
                corr_outputs["correspondence_matrix_mat_path"],
            )
            trace_paths = corr_outputs.get("trace_matrix_paths")
            trace_mat_paths = corr_outputs.get("trace_matrix_mat_paths")
            if trace_paths and trace_mat_paths:
                logger.info(
                    "Trace matrices: %s (npy), %s (mat)",
                    trace_paths,
                    trace_mat_paths,
                )
            else:
                logger.info(
                    "Trace matrix: %s (npy), %s (mat)",
                    corr_outputs["trace_matrix_path"],
                    corr_outputs["trace_matrix_mat_path"],
                )

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
